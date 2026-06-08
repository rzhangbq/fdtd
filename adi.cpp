
#include "adi.H"
#include "init.H"
#include "util.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>

#include <cmath>

using namespace amrex;

namespace
{
    MultiFab makeRhsLike(MultiFab const &field)
    {
        return MultiFab(field.boxArray(), field.DistributionMap(), 1, 0);
    }

    void definePencilFields(ADI::FieldArray &pencils,
                            ADI::FieldArray const &fields,
                            BoxArray const &base_ba,
                            DistributionMapping const &dm)
    {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            BoxArray ba(base_ba);
            ba.convert(fields[idim].ixType());
            pencils[idim].define(ba, dm, fields[idim].nComp(), fields[idim].nGrowVect());
        }
    }

    void copyFields(ADI::FieldArray &dst,
                    ADI::FieldArray const &src,
                    Periodicity const &period)
    {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            dst[idim].ParallelCopy(src[idim], 0, 0, src[idim].nComp(),
                                   IntVect(0), dst[idim].nGrowVect(), period);
        }
    }

    Array<MultiFab, AMREX_SPACEDIM> copyFieldsWithGhosts(
        Array<MultiFab, AMREX_SPACEDIM> const &fields)
    {
        Array<MultiFab, AMREX_SPACEDIM> copies;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            copies[idim].define(fields[idim].boxArray(), fields[idim].DistributionMap(),
                                fields[idim].nComp(), fields[idim].nGrowVect());
            MultiFab::Copy(copies[idim], fields[idim], 0, 0, fields[idim].nComp(),
                           fields[idim].nGrowVect());
        }
        return copies;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void solveTridiagonal(Real const *a,
                          Real const *b,
                          Real const *c,
                          Real const *rhs,
                          Real *x,
                          Real *cprime,
                          Real *dprime,
                          int n)
    {
        for (int i = 0; i < n; ++i)
        {
            dprime[i] = rhs[i];
        }
        Real denom = b[0];
        cprime[0] = c[0] / denom;
        dprime[0] /= denom;

        for (int i = 1; i < n; ++i)
        {
            denom = b[i] - a[i] * cprime[i - 1];
            cprime[i] = (i < n - 1) ? c[i] / denom : 0.0_rt;
            dprime[i] = (dprime[i] - a[i] * dprime[i - 1]) / denom;
        }

        x[n - 1] = dprime[n - 1];
        for (int i = n - 2; i >= 0; --i)
        {
            x[i] = dprime[i] - cprime[i] * x[i + 1];
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void solveCyclicTridiagonal(Real const *a,
                                Real const *b,
                                Real const *c,
                                Real alpha,
                                Real beta,
                                Real const *rhs,
                                Real *x,
                                Real *bb,
                                Real *u,
                                Real *z,
                                Real *cprime,
                                Real *dprime,
                                int n)
    {
        for (int i = 0; i < n; ++i)
        {
            bb[i] = b[i];
            u[i] = 0.0_rt;
        }
        Real const gamma = -b[0];

        bb[0] -= gamma;
        bb[n - 1] -= alpha * beta / gamma;

        u[0] = gamma;
        u[n - 1] = alpha;

        solveTridiagonal(a, bb, c, rhs, x, cprime, dprime, n);
        solveTridiagonal(a, bb, c, u, z, cprime, dprime, n);

        // Applying Sherman-Morrison formula
        // To solve Ax = b
        // Tx = b, Tz = u
        // x -= zv'x / (1 + v'z)
        Real const denom = 1.0_rt + z[0] + beta * z[n - 1] / gamma;
        Real const fact = (x[0] + beta * x[n - 1] / gamma) / denom;

        for (int i = 0; i < n; ++i)
        {
            x[i] -= fact * z[i];
        }
    }

    void solvePeriodicNodalLines(MultiFab &field,
                                 MultiFab const &rhs,
                                 int solve_dir,
                                 Real diag,
                                 std::string const &solver_name)
    {
        Box const domain = field.boxArray().minimalBox();
        int const lo = domain.smallEnd(solve_dir);
        int const hi = domain.bigEnd(solve_dir);
        int const nsolve = hi - lo; // Nodal endpoint at hi duplicates the periodic node at lo.

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            nsolve > 2,
            (solver_name + " requires at least three unique points along the implicit direction")
                .c_str());

        Gpu::DeviceVector<Real> da(nsolve);
        Gpu::DeviceVector<Real> db(nsolve);
        Gpu::DeviceVector<Real> dc(nsolve);
        Real *a = da.data();
        Real *b = db.data();
        Real *c = dc.data();
        ParallelFor(nsolve, [=] AMREX_GPU_DEVICE(int i) noexcept
        {
            a[i] = (i == 0) ? 0.0_rt : -1.0_rt;
            b[i] = diag;
            c[i] = (i == nsolve - 1) ? 0.0_rt : -1.0_rt;
        });

        field.ParallelCopy(rhs, 0, 0, 1);

        for (MFIter mfi(field); mfi.isValid(); ++mfi)
        {
            Box const &bx = mfi.validbox();
            amrex::ignore_unused(solver_name);
            auto const &field_arr = field.array(mfi);

            if (solve_dir == 0)
            {
                auto const b2d = amrex::makeSlab(bx, 0, lo);
                int const jlo = bx.smallEnd(1);
                int const ny = bx.length(1);
                int const klo = bx.smallEnd(2);
                Long const nlines = b2d.numPts();
                Gpu::DeviceVector<Real> d_rhs(nlines * nsolve);
                Gpu::DeviceVector<Real> d_sol(nlines * nsolve);
                Gpu::DeviceVector<Real> d_bb(nlines * nsolve);
                Gpu::DeviceVector<Real> d_u(nlines * nsolve);
                Gpu::DeviceVector<Real> d_z(nlines * nsolve);
                Gpu::DeviceVector<Real> d_cprime(nlines * nsolve);
                Gpu::DeviceVector<Real> d_dprime(nlines * nsolve);
                Real *line_rhs_all = d_rhs.data();
                Real *line_sol_all = d_sol.data();
                Real *line_bb_all = d_bb.data();
                Real *line_u_all = d_u.data();
                Real *line_z_all = d_z.data();
                Real *line_cprime_all = d_cprime.data();
                Real *line_dprime_all = d_dprime.data();

                amrex::ParallelFor(b2d, [=] AMREX_GPU_DEVICE(int, int j, int k) noexcept
                {
                    Long const line = static_cast<Long>(k - klo) * ny + (j - jlo);
                    Long const offset = line * nsolve;
                    Real *line_rhs = line_rhs_all + offset;
                    Real *line_sol = line_sol_all + offset;
                    Real *line_bb = line_bb_all + offset;
                    Real *line_u = line_u_all + offset;
                    Real *line_z = line_z_all + offset;
                    Real *line_cprime = line_cprime_all + offset;
                    Real *line_dprime = line_dprime_all + offset;

                    for (int i = 0; i < nsolve; ++i)
                    {
                        line_rhs[i] = field_arr(lo + i, j, k, 0);
                    }

                    solveCyclicTridiagonal(a, b, c, -1.0_rt, -1.0_rt, line_rhs, line_sol,
                                           line_bb, line_u, line_z, line_cprime, line_dprime, nsolve);

                    for (int i = 0; i < nsolve; ++i)
                    {
                        field_arr(lo + i, j, k, 0) = line_sol[i];
                    }
                    field_arr(hi, j, k, 0) = line_sol[0];
                });
            }
            else if (solve_dir == 1)
            {
                auto const b2d = amrex::makeSlab(bx, 1, lo);
                int const ilo = bx.smallEnd(0);
                int const nx = bx.length(0);
                int const klo = bx.smallEnd(2);
                Long const nlines = b2d.numPts();
                Gpu::DeviceVector<Real> d_rhs(nlines * nsolve);
                Gpu::DeviceVector<Real> d_sol(nlines * nsolve);
                Gpu::DeviceVector<Real> d_bb(nlines * nsolve);
                Gpu::DeviceVector<Real> d_u(nlines * nsolve);
                Gpu::DeviceVector<Real> d_z(nlines * nsolve);
                Gpu::DeviceVector<Real> d_cprime(nlines * nsolve);
                Gpu::DeviceVector<Real> d_dprime(nlines * nsolve);
                Real *line_rhs_all = d_rhs.data();
                Real *line_sol_all = d_sol.data();
                Real *line_bb_all = d_bb.data();
                Real *line_u_all = d_u.data();
                Real *line_z_all = d_z.data();
                Real *line_cprime_all = d_cprime.data();
                Real *line_dprime_all = d_dprime.data();

                amrex::ParallelFor(b2d, [=] AMREX_GPU_DEVICE(int i, int, int k) noexcept
                {
                    Long const line = static_cast<Long>(k - klo) * nx + (i - ilo);
                    Long const offset = line * nsolve;
                    Real *line_rhs = line_rhs_all + offset;
                    Real *line_sol = line_sol_all + offset;
                    Real *line_bb = line_bb_all + offset;
                    Real *line_u = line_u_all + offset;
                    Real *line_z = line_z_all + offset;
                    Real *line_cprime = line_cprime_all + offset;
                    Real *line_dprime = line_dprime_all + offset;

                    for (int j = 0; j < nsolve; ++j)
                    {
                        line_rhs[j] = field_arr(i, lo + j, k, 0);
                    }

                    solveCyclicTridiagonal(a, b, c, -1.0_rt, -1.0_rt, line_rhs, line_sol,
                                           line_bb, line_u, line_z, line_cprime, line_dprime, nsolve);

                    for (int j = 0; j < nsolve; ++j)
                    {
                        field_arr(i, lo + j, k, 0) = line_sol[j];
                    }
                    field_arr(i, hi, k, 0) = line_sol[0];
                });
            }
            else if (solve_dir == 2)
            {
                auto const b2d = amrex::makeSlab(bx, 2, lo);
                int const ilo = bx.smallEnd(0);
                int const nx = bx.length(0);
                int const jlo = bx.smallEnd(1);
                Long const nlines = b2d.numPts();
                Gpu::DeviceVector<Real> d_rhs(nlines * nsolve);
                Gpu::DeviceVector<Real> d_sol(nlines * nsolve);
                Gpu::DeviceVector<Real> d_bb(nlines * nsolve);
                Gpu::DeviceVector<Real> d_u(nlines * nsolve);
                Gpu::DeviceVector<Real> d_z(nlines * nsolve);
                Gpu::DeviceVector<Real> d_cprime(nlines * nsolve);
                Gpu::DeviceVector<Real> d_dprime(nlines * nsolve);
                Real *line_rhs_all = d_rhs.data();
                Real *line_sol_all = d_sol.data();
                Real *line_bb_all = d_bb.data();
                Real *line_u_all = d_u.data();
                Real *line_z_all = d_z.data();
                Real *line_cprime_all = d_cprime.data();
                Real *line_dprime_all = d_dprime.data();

                amrex::ParallelFor(b2d, [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept
                {
                    Long const line = static_cast<Long>(j - jlo) * nx + (i - ilo);
                    Long const offset = line * nsolve;
                    Real *line_rhs = line_rhs_all + offset;
                    Real *line_sol = line_sol_all + offset;
                    Real *line_bb = line_bb_all + offset;
                    Real *line_u = line_u_all + offset;
                    Real *line_z = line_z_all + offset;
                    Real *line_cprime = line_cprime_all + offset;
                    Real *line_dprime = line_dprime_all + offset;

                    for (int k = 0; k < nsolve; ++k)
                    {
                        line_rhs[k] = field_arr(i, j, lo + k, 0);
                    }

                    solveCyclicTridiagonal(a, b, c, -1.0_rt, -1.0_rt, line_rhs, line_sol,
                                           line_bb, line_u, line_z, line_cprime, line_dprime, nsolve);

                    for (int k = 0; k < nsolve; ++k)
                    {
                        field_arr(i, j, lo + k, 0) = line_sol[k];
                    }
                    field_arr(i, j, hi, 0) = line_sol[0];
                });
            }
            else
            {
                amrex::Abort("solve_dir must be 0, 1, or 2");
            }
        }

    }
} // namespace

ADI::ADI()
{
    ParmParse pp("adi");
    pp.getarr("n_cells", m_n_cells);
    pp.query("max_grid_size", m_max_grid_size);

    RealVect prob_lo, prob_hi;
    pp.getarr("prob_lo", prob_lo);
    pp.getarr("prob_hi", prob_hi);

    pp.query("max_step", m_max_step);
    pp.query("plot_int", m_plot_int);
    pp.query("plot_format", m_plot_format);
    pp.query("cfl", m_cfl);
    pp.query("output_dir", m_output_dir);

    if (m_plot_format != "numpy" && m_plot_format != "visit")
    {
        amrex::Abort("adi.plot_format must be \"numpy\" or \"visit\"");
    }

    pp.query("ic", m_ic);
    pp.query("ic_amplitude", m_ic_amplitude);
    pp.query("ic_dir", m_ic_dir);
    pp.query("ic_pol", m_ic_pol);
    pp.query("ic_wavelength", m_ic_wavelength);

    m_pulse_center = 0.5_rt * (prob_lo[m_ic_dir] + prob_hi[m_ic_dir]);
    pp.query("pulse_center", m_pulse_center);
    pp.query("pulse_sigma", m_pulse_sigma);

    Box domain(IntVect(0), m_n_cells - 1);
    RealBox real_box(prob_lo.begin(), prob_hi.begin());
    Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1, 1, 1)};

    m_geom.define(domain, real_box, CoordSys::cartesian, is_periodic);

    m_grids.define(domain);
    m_grids.maxSize(m_max_grid_size);

    m_dmap.define(m_grids);

    static_assert(AMREX_SPACEDIM == 3, "3D only");
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        IntVect etyp(1); // nodal by default
        etyp[idim] = 0;  // cell-centered in idim-direction
        m_efields[idim].define(amrex::convert(m_grids, etyp), m_dmap,
                               1, 1); // one component, one ghost
        IntVect btyp(0);              // cell-centerd by default
        btyp[idim] = 1;               // nodal in idim-direction
        m_bfields[idim].define(amrex::convert(m_grids, btyp), m_dmap, 1, 1);
    }
}

void ADI::initData()
{
    InitSetupFields("adi", m_ic, m_ic_amplitude, m_ic_dir,
                    m_ic_pol, m_ic_wavelength, m_pulse_center, m_pulse_sigma,
                    m_geom, m_efields, m_bfields);
}

void ADI::evolve()
{
    constexpr Real c = 2.99792458e8;

    auto dxinv = m_geom.InvCellSizeArray();
    Real inv_dx2_sum = 0.0_rt;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        inv_dx2_sum += dxinv[idim] * dxinv[idim];
    }
    Real dt = m_cfl / (c * std::sqrt(inv_dx2_sum));

    Real time = 0.0_rt;

    Box const &domain = m_geom.Domain();
    int const nprocs = ParallelDescriptor::NProcs();
    BoxArray bax = amrex::decompose(domain, nprocs, {false, true, true});
    BoxArray bay = amrex::decompose(domain, nprocs, {true, false, true});
    BoxArray baz = amrex::decompose(domain, nprocs, {true, true, false});

    DistributionMapping dmx(bax);
    DistributionMapping dmy(bay);
    DistributionMapping dmz(baz);

    FieldArray efields_x, efields_y, efields_z;
    FieldArray bfields_x, bfields_y, bfields_z;
    definePencilFields(efields_x, m_efields, bax, dmx);
    definePencilFields(efields_y, m_efields, bay, dmy);
    definePencilFields(efields_z, m_efields, baz, dmz);
    definePencilFields(bfields_x, m_bfields, bax, dmx);
    definePencilFields(bfields_y, m_bfields, bay, dmy);
    definePencilFields(bfields_z, m_bfields, baz, dmz);

    // Initialize ghost values once before stepping.
    {
        auto const period = m_geom.periodicity();
        Vector<MultiFab *> efield_ptrs{AMREX_D_DECL(&m_efields[0], &m_efields[1], &m_efields[2])};
        Vector<MultiFab *> bfield_ptrs{AMREX_D_DECL(&m_bfields[0], &m_bfields[1], &m_bfields[2])};
        amrex::FillBoundary(efield_ptrs, period);
        amrex::FillBoundary(bfield_ptrs, period);
    }

    if (m_plot_int > 0)
    {
        UtilWritePlotOutput(m_plot_format, m_output_dir, 0, time,
                            m_grids, m_dmap, m_geom, m_efields, m_bfields);
    }

    for (int step = 0; step < m_max_step; ++step)
    {
        adiFirstHalfStep(m_efields, m_bfields, efields_x, efields_y,
                         efields_z, bfields_x, bfields_y, bfields_z, dt);
        adiSecondHalfStep(m_efields, m_bfields, efields_x, efields_y,
                          efields_z, bfields_x, bfields_y, bfields_z, dt);

        time += dt;

        if (m_plot_int > 0 && (step + 1) % m_plot_int == 0)
        {
            UtilWritePlotOutput(m_plot_format, m_output_dir, step + 1, time,
                                m_grids, m_dmap, m_geom, m_efields, m_bfields);
        }
    }
}

void ADI::adiFirstHalfStep(Array<MultiFab, AMREX_SPACEDIM>& efields,
                           Array<MultiFab, AMREX_SPACEDIM>& bfields,
                           FieldArray& efields_x, FieldArray& efields_y,
                           FieldArray& efields_z, FieldArray& bfields_x,
                           FieldArray& bfields_y, FieldArray& bfields_z,
                           Real dt)
{
    // eq:adi-first-half-amrex — implicit E along y,z,x; explicit B at n+1/2
    auto const period = m_geom.periodicity();
    Vector<MultiFab *> efield_ptrs{AMREX_D_DECL(&efields[0], &efields[1], &efields[2])};
    Vector<MultiFab *> bfield_ptrs{AMREX_D_DECL(&bfields[0], &bfields[1], &bfields[2])};

    Array<MultiFab, AMREX_SPACEDIM> eold = copyFieldsWithGhosts(efields);

    copyFields(efields_y, efields, period);
    copyFields(bfields_y, bfields, period);
    MultiFab rhs_ex = buildRhsEx1(efields_y, bfields_y, dt);

    copyFields(efields_z, efields, period);
    copyFields(bfields_z, bfields, period);
    MultiFab rhs_ey = buildRhsEy1(efields_z, bfields_z, dt);

    copyFields(efields_x, efields, period);
    copyFields(bfields_x, bfields, period);
    MultiFab rhs_ez = buildRhsEz1(efields_x, bfields_x, dt);

    solveImplicitEx1(efields_y[0], rhs_ex, dt);
    solveImplicitEy1(efields_z[1], rhs_ey, dt);
    solveImplicitEz1(efields_x[2], rhs_ez, dt);

    efields[0].ParallelCopy(efields_y[0], 0, 0, 1,
                            IntVect(0), IntVect(0), period);
    efields[1].ParallelCopy(efields_z[1], 0, 0, 1,
                            IntVect(0), IntVect(0), period);
    efields[2].ParallelCopy(efields_x[2], 0, 0, 1,
                            IntVect(0), IntVect(0), period);

    amrex::FillBoundary(efield_ptrs, period);

    stepBx(bfields[0], efields[1], eold[2], dt);
    stepBy(bfields[1], efields[2], eold[0], dt);
    stepBz(bfields[2], efields[0], eold[1], dt);

    amrex::FillBoundary(bfield_ptrs, period);
}

void ADI::adiSecondHalfStep(Array<MultiFab, AMREX_SPACEDIM>& efields,
                            Array<MultiFab, AMREX_SPACEDIM>& bfields,
                            FieldArray& efields_x, FieldArray& efields_y,
                            FieldArray& efields_z, FieldArray& bfields_x,
                            FieldArray& bfields_y, FieldArray& bfields_z,
                            Real dt)
{
    // eq:adi-second-half-amrex — implicit E along z,x,y; explicit B at n+1
    auto const period = m_geom.periodicity();
    Vector<MultiFab *> efield_ptrs{AMREX_D_DECL(&efields[0], &efields[1], &efields[2])};
    Vector<MultiFab *> bfield_ptrs{AMREX_D_DECL(&bfields[0], &bfields[1], &bfields[2])};

    Array<MultiFab, AMREX_SPACEDIM> eold = copyFieldsWithGhosts(efields);

    copyFields(efields_z, efields, period);
    copyFields(bfields_z, bfields, period);
    MultiFab rhs_ex = buildRhsEx2(efields_z, bfields_z, dt);

    copyFields(efields_x, efields, period);
    copyFields(bfields_x, bfields, period);
    MultiFab rhs_ey = buildRhsEy2(efields_x, bfields_x, dt);

    copyFields(efields_y, efields, period);
    copyFields(bfields_y, bfields, period);
    MultiFab rhs_ez = buildRhsEz2(efields_y, bfields_y, dt);

    solveImplicitEx2(efields_z[0], rhs_ex, dt);
    solveImplicitEy2(efields_x[1], rhs_ey, dt);
    solveImplicitEz2(efields_y[2], rhs_ez, dt);

    efields[0].ParallelCopy(efields_z[0], 0, 0, 1,
                            IntVect(0), IntVect(0), period);
    efields[1].ParallelCopy(efields_x[1], 0, 0, 1,
                            IntVect(0), IntVect(0), period);
    efields[2].ParallelCopy(efields_y[2], 0, 0, 1,
                            IntVect(0), IntVect(0), period);

    amrex::FillBoundary(efield_ptrs, period);

    stepBx(bfields[0], eold[1], efields[2], dt);
    stepBy(bfields[1], eold[2], efields[0], dt);
    stepBz(bfields[2], eold[0], efields[1], dt);

    amrex::FillBoundary(bfield_ptrs, period);
}

MultiFab ADI::buildRhsEx1(Array<MultiFab, AMREX_SPACEDIM> const& efields,
                          Array<MultiFab, AMREX_SPACEDIM> const& bfields,
                          Real dt) const
{
    // RHS of eq:adi-first-half-amrex Ex row (vacuum), for tridiagonal solve along y.
    constexpr Real c = 2.99792458e8;

    MultiFab rhs = makeRhsLike(efields[0]);

    auto const dx = m_geom.CellSizeArray();
    Real const dy = dx[1];
    Real const dyinv = 1.0_rt / dy;
    Real const dzinv = 1.0_rt / dx[2];
    Real const dxinv = 1.0_rt / dx[0];

    Real const coef_ex = 4.0_rt * dy * dy / (c * c * dt * dt);
    Real const coef_b = 2.0_rt * dy * dy / dt;
    Real const coef_ey = dy;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box &tilebx = mfi.tilebox();
        auto const &rhs_arr = rhs.array(mfi);
        auto const &ex_arr = efields[0].const_array(mfi);
        auto const &ey_arr = efields[1].const_array(mfi);
        auto const &bz_arr = bfields[2].const_array(mfi);
        auto const &by_arr = bfields[1].const_array(mfi);

        ParallelFor(tilebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                      {
            Real const curl_b = dyinv * (bz_arr(i, j, k) - bz_arr(i, j - 1, k)) -
                                dzinv * (by_arr(i, j, k) - by_arr(i, j, k - 1));
            Real const dey_dx = dxinv * ((ey_arr(i + 1, j - 1, k) - ey_arr(i, j - 1, k)) -
                                         (ey_arr(i + 1, j, k) - ey_arr(i, j, k)));
            rhs_arr(i, j, k) = coef_ex * ex_arr(i, j, k) + coef_b * curl_b + coef_ey * dey_dx;
        });
    }

    return rhs;
}

MultiFab ADI::buildRhsEy1(Array<MultiFab, AMREX_SPACEDIM> const& efields,
                          Array<MultiFab, AMREX_SPACEDIM> const& bfields,
                          Real dt) const
{
    // RHS of eq:adi-first-half-amrex Ey row (vacuum), for tridiagonal solve along z.
    constexpr Real c = 2.99792458e8;

    MultiFab rhs = makeRhsLike(efields[1]);

    auto const dx = m_geom.CellSizeArray();
    Real const dz = dx[2];
    Real const dzinv = 1.0_rt / dz;
    Real const dxinv = 1.0_rt / dx[0];
    Real const dyinv = 1.0_rt / dx[1];

    Real const coef_ey = 4.0_rt * dz * dz / (c * c * dt * dt);
    Real const coef_b = 2.0_rt * dz * dz / dt;
    Real const coef_ez = dz;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box &tilebx = mfi.tilebox();
        auto const &rhs_arr = rhs.array(mfi);
        auto const &ey_arr = efields[1].const_array(mfi);
        auto const &ez_arr = efields[2].const_array(mfi);
        auto const &bx_arr = bfields[0].const_array(mfi);
        auto const &bz_arr = bfields[2].const_array(mfi);

        ParallelFor(tilebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                      {
            Real const curl_b = dzinv * (bx_arr(i, j, k) - bx_arr(i, j, k - 1)) -
                                dxinv * (bz_arr(i, j, k) - bz_arr(i - 1, j, k));
            Real const dez_dy = dyinv * ((ez_arr(i, j + 1, k - 1) - ez_arr(i, j, k - 1)) -
                                         (ez_arr(i, j + 1, k) - ez_arr(i, j, k)));
            rhs_arr(i, j, k) = coef_ey * ey_arr(i, j, k) + coef_b * curl_b + coef_ez * dez_dy;
        });
    }

    return rhs;
}

MultiFab ADI::buildRhsEz1(Array<MultiFab, AMREX_SPACEDIM> const& efields,
                          Array<MultiFab, AMREX_SPACEDIM> const& bfields,
                          Real dt) const
{
    // RHS of eq:adi-first-half-amrex Ez row (vacuum), for tridiagonal solve along x.
    constexpr Real c = 2.99792458e8;

    MultiFab rhs = makeRhsLike(efields[2]);

    auto const dx = m_geom.CellSizeArray();
    Real const dx_cell = dx[0];
    Real const dxinv = 1.0_rt / dx_cell;
    Real const dyinv = 1.0_rt / dx[1];
    Real const dzinv = 1.0_rt / dx[2];

    Real const coef_ez = 4.0_rt * dx_cell * dx_cell / (c * c * dt * dt);
    Real const coef_b = 2.0_rt * dx_cell * dx_cell / dt;
    Real const coef_ex = dx_cell;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box &tilebx = mfi.tilebox();
        auto const &rhs_arr = rhs.array(mfi);
        auto const &ez_arr = efields[2].const_array(mfi);
        auto const &ex_arr = efields[0].const_array(mfi);
        auto const &by_arr = bfields[1].const_array(mfi);
        auto const &bx_arr = bfields[0].const_array(mfi);

        ParallelFor(tilebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                      {
            Real const curl_b = dxinv * (by_arr(i, j, k) - by_arr(i - 1, j, k)) -
                                dyinv * (bx_arr(i, j, k) - bx_arr(i, j - 1, k));
            Real const dex_dz = dzinv * ((ex_arr(i - 1, j, k + 1) - ex_arr(i - 1, j, k)) -
                                         (ex_arr(i, j, k + 1) - ex_arr(i, j, k)));
            rhs_arr(i, j, k) = coef_ez * ez_arr(i, j, k) + coef_b * curl_b + coef_ex * dex_dz;
        });
    }

    return rhs;
}

void ADI::solveImplicitEx1(MultiFab &ex, MultiFab const &rhs, Real dt) const
{
    constexpr Real c = 2.99792458e8;
    Real const dy = m_geom.CellSizeArray()[1];
    Real const diag = 2.0_rt + 4.0_rt * dy * dy / (c * c * dt * dt);
    solvePeriodicNodalLines(ex, rhs, 1, diag, "solveImplicitEx1");
}

void ADI::solveImplicitEy1(MultiFab &ey, MultiFab const &rhs, Real dt) const
{
    constexpr Real c = 2.99792458e8;
    Real const dz = m_geom.CellSizeArray()[2];
    Real const diag = 2.0_rt + 4.0_rt * dz * dz / (c * c * dt * dt);
    solvePeriodicNodalLines(ey, rhs, 2, diag, "solveImplicitEy1");
}

void ADI::solveImplicitEz1(MultiFab &ez, MultiFab const &rhs, Real dt) const
{
    constexpr Real c = 2.99792458e8;
    Real const dx = m_geom.CellSizeArray()[0];
    Real const diag = 2.0_rt + 4.0_rt * dx * dx / (c * c * dt * dt);
    solvePeriodicNodalLines(ez, rhs, 0, diag, "solveImplicitEz1");
}

MultiFab ADI::buildRhsEx2(Array<MultiFab, AMREX_SPACEDIM> const& efields,
                          Array<MultiFab, AMREX_SPACEDIM> const& bfields,
                          Real dt) const
{
    // RHS of eq:adi-second-half-amrex Ex row (vacuum), for tridiagonal solve along z.
    constexpr Real c = 2.99792458e8;

    MultiFab rhs = makeRhsLike(efields[0]);

    auto const dx = m_geom.CellSizeArray();
    Real const dz = dx[2];
    Real const dyinv = 1.0_rt / dx[1];
    Real const dzinv = 1.0_rt / dz;
    Real const dxinv = 1.0_rt / dx[0];

    Real const coef_ex = 4.0_rt * dz * dz / (c * c * dt * dt);
    Real const coef_b = 2.0_rt * dz * dz / dt;
    Real const coef_ez = dz;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box &tilebx = mfi.tilebox();
        auto const &rhs_arr = rhs.array(mfi);
        auto const &ex_arr = efields[0].const_array(mfi);
        auto const &ez_arr = efields[2].const_array(mfi);
        auto const &bz_arr = bfields[2].const_array(mfi);
        auto const &by_arr = bfields[1].const_array(mfi);

        ParallelFor(tilebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                      {
            Real const curl_b = dyinv * (bz_arr(i, j, k) - bz_arr(i, j - 1, k)) -
                                dzinv * (by_arr(i, j, k) - by_arr(i, j, k - 1));
            Real const dez_dx = dxinv * ((ez_arr(i + 1, j, k - 1) - ez_arr(i, j, k - 1)) -
                                         (ez_arr(i + 1, j, k) - ez_arr(i, j, k)));
            rhs_arr(i, j, k) = coef_ex * ex_arr(i, j, k) + coef_b * curl_b + coef_ez * dez_dx;
        });
    }

    return rhs;
}

MultiFab ADI::buildRhsEy2(Array<MultiFab, AMREX_SPACEDIM> const& efields,
                          Array<MultiFab, AMREX_SPACEDIM> const& bfields,
                          Real dt) const
{
    // RHS of eq:adi-second-half-amrex Ey row (vacuum), for tridiagonal solve along x.
    constexpr Real c = 2.99792458e8;

    MultiFab rhs = makeRhsLike(efields[1]);

    auto const dx = m_geom.CellSizeArray();
    Real const dx_cell = dx[0];
    Real const dzinv = 1.0_rt / dx[2];
    Real const dxinv = 1.0_rt / dx_cell;
    Real const dyinv = 1.0_rt / dx[1];

    Real const coef_ey = 4.0_rt * dx_cell * dx_cell / (c * c * dt * dt);
    Real const coef_b = 2.0_rt * dx_cell * dx_cell / dt;
    Real const coef_ex = dx_cell;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box &tilebx = mfi.tilebox();
        auto const &rhs_arr = rhs.array(mfi);
        auto const &ey_arr = efields[1].const_array(mfi);
        auto const &ex_arr = efields[0].const_array(mfi);
        auto const &bx_arr = bfields[0].const_array(mfi);
        auto const &bz_arr = bfields[2].const_array(mfi);

        ParallelFor(tilebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                      {
            Real const curl_b = dzinv * (bx_arr(i, j, k) - bx_arr(i, j, k - 1)) -
                                dxinv * (bz_arr(i, j, k) - bz_arr(i - 1, j, k));
            Real const dex_dy = dyinv * ((ex_arr(i - 1, j + 1, k) - ex_arr(i - 1, j, k)) -
                                         (ex_arr(i, j + 1, k) - ex_arr(i, j, k)));
            rhs_arr(i, j, k) = coef_ey * ey_arr(i, j, k) + coef_b * curl_b + coef_ex * dex_dy;
        });
    }

    return rhs;
}

MultiFab ADI::buildRhsEz2(Array<MultiFab, AMREX_SPACEDIM> const& efields,
                          Array<MultiFab, AMREX_SPACEDIM> const& bfields,
                          Real dt) const
{
    // RHS of eq:adi-second-half-amrex Ez row (vacuum), for tridiagonal solve along y.
    constexpr Real c = 2.99792458e8;

    MultiFab rhs = makeRhsLike(efields[2]);

    auto const dx = m_geom.CellSizeArray();
    Real const dy = dx[1];
    Real const dxinv = 1.0_rt / dx[0];
    Real const dyinv = 1.0_rt / dy;
    Real const dzinv = 1.0_rt / dx[2];

    Real const coef_ez = 4.0_rt * dy * dy / (c * c * dt * dt);
    Real const coef_b = 2.0_rt * dy * dy / dt;
    Real const coef_ey = dy;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box &tilebx = mfi.tilebox();
        auto const &rhs_arr = rhs.array(mfi);
        auto const &ez_arr = efields[2].const_array(mfi);
        auto const &ey_arr = efields[1].const_array(mfi);
        auto const &by_arr = bfields[1].const_array(mfi);
        auto const &bx_arr = bfields[0].const_array(mfi);

        ParallelFor(tilebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                      {
            Real const curl_b = dxinv * (by_arr(i, j, k) - by_arr(i - 1, j, k)) -
                                dyinv * (bx_arr(i, j, k) - bx_arr(i, j - 1, k));
            Real const dey_dz = dzinv * ((ey_arr(i, j - 1, k + 1) - ey_arr(i, j - 1, k)) -
                                         (ey_arr(i, j, k + 1) - ey_arr(i, j, k)));
            rhs_arr(i, j, k) = coef_ez * ez_arr(i, j, k) + coef_b * curl_b + coef_ey * dey_dz;
        });
    }

    return rhs;
}

void ADI::solveImplicitEx2(MultiFab &ex, MultiFab const &rhs, Real dt) const
{
    constexpr Real c = 2.99792458e8;
    Real const dz = m_geom.CellSizeArray()[2];
    Real const diag = 2.0_rt + 4.0_rt * dz * dz / (c * c * dt * dt);
    solvePeriodicNodalLines(ex, rhs, 2, diag, "solveImplicitEx2");
}

void ADI::solveImplicitEy2(MultiFab &ey, MultiFab const &rhs, Real dt) const
{
    constexpr Real c = 2.99792458e8;
    Real const dx = m_geom.CellSizeArray()[0];
    Real const diag = 2.0_rt + 4.0_rt * dx * dx / (c * c * dt * dt);
    solvePeriodicNodalLines(ey, rhs, 0, diag, "solveImplicitEy2");
}

void ADI::solveImplicitEz2(MultiFab &ez, MultiFab const &rhs, Real dt) const
{
    constexpr Real c = 2.99792458e8;
    Real const dy = m_geom.CellSizeArray()[1];
    Real const diag = 2.0_rt + 4.0_rt * dy * dy / (c * c * dt * dt);
    solvePeriodicNodalLines(ez, rhs, 1, diag, "solveImplicitEz2");
}

void ADI::stepBx(MultiFab &bx_dst, MultiFab const &ey_src,
                 MultiFab const &ez_src, Real dt)
{
    // B_x += (dt/2)(dEy/dz - dEz/dy), vacuum Yee stencil.
    auto const dxinv = m_geom.InvCellSizeArray();
    Real const halfdt = 0.5_rt * dt;

    auto const &ey = ey_src.const_arrays();
    auto const &ez = ez_src.const_arrays();
    auto const &bx = bx_dst.arrays();

    ParallelFor(bx_dst, [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                {
        bx[b](i, j, k) +=
            halfdt * (dxinv[2] * (ey[b](i, j, k + 1) - ey[b](i, j, k)) -
                      dxinv[1] * (ez[b](i, j + 1, k) - ez[b](i, j, k)));
    });
    Gpu::streamSynchronize();
}

void ADI::stepBy(MultiFab &by_dst, MultiFab const &ez_src,
                 MultiFab const &ex_src, Real dt)
{
    auto const dxinv = m_geom.InvCellSizeArray();
    Real const halfdt = 0.5_rt * dt;

    auto const &ex = ex_src.const_arrays();
    auto const &ez = ez_src.const_arrays();
    auto const &by = by_dst.arrays();

    ParallelFor(by_dst, [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                {
        by[b](i, j, k) +=
            halfdt * (dxinv[0] * (ez[b](i + 1, j, k) - ez[b](i, j, k)) -
                      dxinv[2] * (ex[b](i, j, k + 1) - ex[b](i, j, k)));
    });
    Gpu::streamSynchronize();
}

void ADI::stepBz(MultiFab &bz_dst, MultiFab const &ex_src,
                 MultiFab const &ey_src, Real dt)
{
    auto const dxinv = m_geom.InvCellSizeArray();
    Real const halfdt = 0.5_rt * dt;

    auto const &ex = ex_src.const_arrays();
    auto const &ey = ey_src.const_arrays();
    auto const &bz = bz_dst.arrays();

    ParallelFor(bz_dst, [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                {
        bz[b](i, j, k) +=
            halfdt * (dxinv[1] * (ex[b](i, j + 1, k) - ex[b](i, j, k)) -
                      dxinv[0] * (ey[b](i + 1, j, k) - ey[b](i, j, k)));
    });
    Gpu::streamSynchronize();
}
