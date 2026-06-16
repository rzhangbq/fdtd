
#include "fdtd.H"
#include "init.H"
#include "pec.H"
#include "util.H"

#include <AMReX_ParmParse.H>

using namespace amrex;

FDTD::FDTD()
{
    ParmParse pp("fdtd");
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
        amrex::Abort("fdtd.plot_format must be \"numpy\" or \"visit\"");
    }

    pp.query("ic", m_ic);
    pp.query("ic_amplitude", m_ic_amplitude);
    pp.query("ic_dir", m_ic_dir);
    pp.query("ic_pol", m_ic_pol);
    pp.query("ic_wavelength", m_ic_wavelength);

    m_pulse_center = 0.5_rt * (prob_lo[m_ic_dir] + prob_hi[m_ic_dir]);
    pp.query("pulse_center", m_pulse_center);
    pp.query("pulse_sigma", m_pulse_sigma);

    m_pec_normal = -1;
    pp.query("pec_normal", m_pec_normal);
    if (m_pec_normal < -1 || m_pec_normal > 2)
    {
        amrex::Abort("fdtd.pec_normal must be -1 (none) or 0, 1, 2");
    }

    Box domain(IntVect(0), m_n_cells - 1);
    RealBox real_box(prob_lo.begin(), prob_hi.begin());
    Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1, 1, 1)};
    if (m_pec_normal >= 0)
    {
        is_periodic[m_pec_normal] = 0;
    }

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

void FDTD::initData()
{
    InitSetupFields("fdtd", m_ic, m_ic_amplitude, m_ic_dir,
                    m_ic_pol, m_ic_wavelength, m_pulse_center, m_pulse_sigma,
                    m_geom, m_efields, m_bfields);
    PecPinTangentialEwalls(m_pec_normal, m_efields);
}

void FDTD::evolve()
{
    constexpr Real c = 2.99792458e8;

    auto dxinv = m_geom.InvCellSizeArray();
    // 3D Yee CFL: dt <= cfl / (c * sqrt(sum_i 1/dx_i^2))  (see notes/adi.tex)
    Real inv_dx2_sum = 0.0_rt;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        inv_dx2_sum += dxinv[idim] * dxinv[idim];
    }
    Real dt = m_cfl / (c * std::sqrt(inv_dx2_sum));
    Real c2dt = c * c * dt;

    auto const period = m_geom.periodicity();
    Vector<MultiFab *> efields{AMREX_D_DECL(&m_efields[0], &m_efields[1], &m_efields[2])};
    Vector<MultiFab *> bfields{AMREX_D_DECL(&m_bfields[0], &m_bfields[1], &m_bfields[2])};

    int const pec = m_pec_normal;
    Box const ex_box = m_efields[0].boxArray().minimalBox();
    Box const ey_box = m_efields[1].boxArray().minimalBox();
    Box const ez_box = m_efields[2].boxArray().minimalBox();
    int const ex_plo = (pec >= 0) ? ex_box.smallEnd(pec) : 0;
    int const ex_phi = (pec >= 0) ? ex_box.bigEnd(pec) : 0;
    int const ey_plo = (pec >= 0) ? ey_box.smallEnd(pec) : 0;
    int const ey_phi = (pec >= 0) ? ey_box.bigEnd(pec) : 0;
    int const ez_plo = (pec >= 0) ? ez_box.smallEnd(pec) : 0;
    int const ez_phi = (pec >= 0) ? ez_box.bigEnd(pec) : 0;
    bool const ex_nodal_on_pec = pec >= 0 && m_efields[0].ixType().nodeCentered(pec);
    bool const ey_nodal_on_pec = pec >= 0 && m_efields[1].ixType().nodeCentered(pec);
    bool const ez_nodal_on_pec = pec >= 0 && m_efields[2].ixType().nodeCentered(pec);

    Real time = 0.0_rt;

    if (m_plot_int > 0)
    {
        UtilWritePlotOutput(m_plot_format, m_output_dir, 0, time,
                            m_grids, m_dmap, m_geom, m_efields, m_bfields);
    }

    for (int step = 0; step < m_max_step; ++step)
    {
        amrex::FillBoundary(efields, period);

        auto const &bx = m_bfields[0].arrays();
        auto const &by = m_bfields[1].arrays();
        auto const &bz = m_bfields[2].arrays();
        auto const &ex = m_efields[0].arrays();
        auto const &ey = m_efields[1].arrays();
        auto const &ez = m_efields[2].arrays();

        Real halfdt = 0.5_rt * dt;
        ParallelFor(m_bfields[0], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                    { bx[b](i, j, k) -= halfdt * (dxinv[1] * (ez[b](i, j + 1, k) - ez[b](i, j, k)) - dxinv[2] * (ey[b](i, j, k + 1) - ey[b](i, j, k))); });
        ParallelFor(m_bfields[1], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                    { by[b](i, j, k) -= halfdt * (dxinv[2] * (ex[b](i, j, k + 1) - ex[b](i, j, k)) - dxinv[0] * (ez[b](i + 1, j, k) - ez[b](i, j, k))); });
        ParallelFor(m_bfields[2], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                    { bz[b](i, j, k) -= halfdt * (dxinv[0] * (ey[b](i + 1, j, k) - ey[b](i, j, k)) - dxinv[1] * (ex[b](i, j + 1, k) - ex[b](i, j, k))); });

        amrex::FillBoundary(bfields, period);

        ParallelFor(m_efields[0], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                    {
            if (PecSkipEupdate(pec, 0, ex_nodal_on_pec, ex_plo, ex_phi, i, j, k))
            {
                ex[b](i, j, k) = 0.0_rt;
                return;
            }
            ex[b](i, j, k) += c2dt * (dxinv[1] * (bz[b](i, j, k) - bz[b](i, j - 1, k)) -
                                        dxinv[2] * (by[b](i, j, k) - by[b](i, j, k - 1))); });
        ParallelFor(m_efields[1], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                    {
            if (PecSkipEupdate(pec, 1, ey_nodal_on_pec, ey_plo, ey_phi, i, j, k))
            {
                ey[b](i, j, k) = 0.0_rt;
                return;
            }
            ey[b](i, j, k) += c2dt * (dxinv[2] * (bx[b](i, j, k) - bx[b](i, j, k - 1)) -
                                        dxinv[0] * (bz[b](i, j, k) - bz[b](i - 1, j, k))); });
        ParallelFor(m_efields[2], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                    {
            if (PecSkipEupdate(pec, 2, ez_nodal_on_pec, ez_plo, ez_phi, i, j, k))
            {
                ez[b](i, j, k) = 0.0_rt;
                return;
            }
            ez[b](i, j, k) += c2dt * (dxinv[0] * (by[b](i, j, k) - by[b](i - 1, j, k)) -
                                        dxinv[1] * (bx[b](i, j, k) - bx[b](i, j - 1, k))); });

        amrex::FillBoundary(efields, period);

        ParallelFor(m_bfields[0], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                    { bx[b](i, j, k) -= halfdt * (dxinv[1] * (ez[b](i, j + 1, k) - ez[b](i, j, k)) - dxinv[2] * (ey[b](i, j, k + 1) - ey[b](i, j, k))); });
        ParallelFor(m_bfields[1], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                    { by[b](i, j, k) -= halfdt * (dxinv[2] * (ex[b](i, j, k + 1) - ex[b](i, j, k)) - dxinv[0] * (ez[b](i + 1, j, k) - ez[b](i, j, k))); });
        ParallelFor(m_bfields[2], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                    { bz[b](i, j, k) -= halfdt * (dxinv[0] * (ey[b](i + 1, j, k) - ey[b](i, j, k)) - dxinv[1] * (ex[b](i, j + 1, k) - ex[b](i, j, k))); });

        time += dt;

        if (m_plot_int > 0 && (step + 1) % m_plot_int == 0)
        {
            UtilWritePlotOutput(m_plot_format, m_output_dir, step + 1, time,
                                m_grids, m_dmap, m_geom, m_efields, m_bfields);
        }
    }
}
