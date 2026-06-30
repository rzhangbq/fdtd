
#include "adi.H"
#include "init.H"
#include "pec.H"
#include "util.H"

#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>

#include <cmath>
#include <vector>

using namespace amrex;

namespace
{
    constexpr Real eps0 = 8.854187817e-12;
    constexpr Real mu0 = 4.0 * M_PI * 1e-7;

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

    // Thomas tridiagonal solve. If rhs != nullptr, solve T x = rhs.
    // If rhs == nullptr, use the sparse Sherman-Morrison RHS u with u[0] = 1 and
    // u[n-1] = alpha/gamma (all other entries zero); alpha and gamma are only read then.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void solveTridiagonal(Real const *a,
                                                                   Real const *b,
                                                                   Real const *c,
                                                                   Real const *rhs,
                                                                   Real *x,
                                                                   Real *cprime,
                                                                   Real *dprime,
                                                                   int n,
                                                                   Real alpha = 0.0_rt,
                                                                   Real gamma = 1.0_rt) noexcept
    {
        AMREX_ALWAYS_ASSERT(n >= 2);

        Real denom = b[0];
        AMREX_ALWAYS_ASSERT(std::abs(denom) > 0.0_rt);
        cprime[0] = c[0] / denom;
        dprime[0] = ((rhs != nullptr) ? rhs[0] : 1.0_rt) / denom;

        for (int i = 1; i < n; ++i)
        {
            denom = b[i] - a[i] * cprime[i - 1];
            AMREX_ALWAYS_ASSERT(std::abs(denom) > 0.0_rt);
            cprime[i] = (i < n - 1) ? c[i] / denom : 0.0_rt;
            Real const rhs_i = (rhs != nullptr) ? rhs[i]
                                                : ((i == n - 1) ? alpha / gamma : 0.0_rt);
            dprime[i] = (rhs_i - a[i] * dprime[i - 1]) / denom;
        }

        x[n - 1] = dprime[n - 1];
        for (int i = n - 2; i >= 0; --i)
        {
            x[i] = dprime[i] - cprime[i] * x[i + 1];
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void solveCyclicTridiagonal(Real const *a,
                                                                         Real const *bb,
                                                                         Real const *c,
                                                                         Real alpha,
                                                                         Real beta,
                                                                         Real gamma,
                                                                         Real const *rhs,
                                                                         Real *x,
                                                                         Real *cprime,
                                                                         Real *dprime,
                                                                         Real *z,
                                                                         int n) noexcept
    {
        AMREX_ALWAYS_ASSERT(n > 2);
        AMREX_ALWAYS_ASSERT(std::abs(gamma) > 0.0_rt);

        // first solve Tx = rhs, with T = [a, bb, c]
        solveTridiagonal(a, bb, c, rhs, x, cprime, dprime, n);

        // second solve Tz = u, u[0] = 1, u[n-1] = alpha/gamma
        solveTridiagonal(a, bb, c, nullptr, z, cprime, dprime, n, alpha, gamma);

        Real const denom = 1.0_rt + gamma * z[0] + beta * z[n - 1];
        AMREX_ALWAYS_ASSERT(std::abs(denom) > 0.0_rt);
        Real const fact = (x[0] + beta * x[n - 1] / gamma) / denom;

        for (int i = 0; i < n; ++i)
        {
            x[i] -= fact * gamma * z[i];
        }
    }

    // Map packed index (0..nsolve-1) to local offset along solve_dir, skipping PEC at iw.
    // Order: 0..iw-1, iw+1..nsolve (no nearest-neighbor link across the gap).
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int packedPecGlobalIdx(int p, int iw) noexcept
    {
        return (p < iw) ? p : p + 1;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int periodicLineGlobalIdx(int p, int iw) noexcept
    {
        return (iw < 0) ? p : packedPecGlobalIdx(p, iw);
    }

    // Periodic cyclic tridiagonal solve along solve_dir with inhomogeneous coefficients.
    // interior_pec_iw < 0: full nodal line (hi duplicates lo); optional tangential PEC skip via pec_* / e_comp.
    // interior_pec_iw >= 0: omit that interior PEC node and break coupling across the gap.
    void solvePeriodicCyclicLines(MultiFab &field,
                                   MultiFab const &rhs,
                                MultiFab const &Cb,
                                MultiFab const &Db,
                                int solve_dir,
                                Real inv_d2,
                                int interior_pec_iw,
                                int pec_normal,
                                int pec_location,
                                int e_comp,
                                std::string const &solver_name)
    {
        Box const domain = field.boxArray().minimalBox();
        int const lo = domain.smallEnd(solve_dir);
        int const hi = domain.bigEnd(solve_dir);
        bool const split_pec = interior_pec_iw >= 0;
        int const nsolve = split_pec ? hi - lo - 1 : hi - lo;
        int const iw = split_pec ? interior_pec_iw - lo : -1;

        bool const skip_pec_lines =
            pec_normal >= 0 && pec_location != 0 && e_comp != pec_normal &&
            solve_dir != pec_normal && field.ixType().nodeCentered(pec_normal);
        int const pec_lo = skip_pec_lines ? domain.smallEnd(pec_normal) : 0;
        int const pec_hi = skip_pec_lines ? domain.bigEnd(pec_normal) : 0;

        if (split_pec)
        {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                iw > 0 && iw < nsolve,
                (solver_name +
                 " requires pec_location strictly interior along the implicit direction")
                    .c_str());
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            nsolve > 2,
            (solver_name + " requires at least three unique points along the implicit direction")
                .c_str());

        field.ParallelCopy(rhs, 0, 0, 1);

        constexpr int n_line_work = 4; // cprime, dprime, x, z
        constexpr int n_line_coeff = 3; // a, bb, c per row

        for (MFIter mfi(field); mfi.isValid(); ++mfi)
        {
            Box const &bx = mfi.validbox();
            amrex::ignore_unused(solver_name);
            auto const &field_arr = field.array(mfi);
            auto const &cb_arr = Cb.const_array(mfi);
            auto const &db_arr = Db.const_array(mfi);

            if (solve_dir == 0)
            {
                Box const b2d = amrex::makeSlab(bx, 0, lo);
                Long const nlines = b2d.numPts();
                Gpu::DeviceVector<Real> line_work(nlines * nsolve * n_line_work);
                Gpu::DeviceVector<Real> line_coeff(nlines * nsolve * n_line_coeff);
                Real *work = line_work.data();
                Real *coeff = line_coeff.data();
                int const jlo = b2d.smallEnd(1);
                int const klo = b2d.smallEnd(2);
                int const jlen = b2d.length(1);

                amrex::ParallelForOMP(b2d, [=] AMREX_GPU_DEVICE(int, int j, int k) noexcept
                                      {
                    if (skip_pec_lines &&
                        PecOnPlane(pec_normal, pec_location, pec_lo, pec_hi, 0, j, k))
                    {
                        for (int ii = lo; ii <= hi; ++ii)
                        {
                            field_arr(ii, j, k) = 0.0_rt;
                        }
                        return;
                    }

                    int const line_id = (j - jlo) + (k - klo) * jlen;
                    Real *cprime = work + line_id * nsolve * n_line_work;
                    Real *dprime = cprime + nsolve;
                    Real *x = dprime + nsolve;
                    Real *z = x + nsolve;
                    Real *a = coeff + line_id * nsolve * n_line_coeff;
                    Real *bb = a + nsolve;
                    Real *c = bb + nsolve;

                    Real db_seam = db_arr(hi - 1, j, k);
                    Real alpha_cyclic = -db_seam * inv_d2;
                    Real beta_cyclic = -db_seam * inv_d2;

                    for (int p = 0; p < nsolve; ++p)
                    {
                        int const g = periodicLineGlobalIdx(p, iw);
                        int const i = lo + g;

                        Real db_lo = 0.0_rt;
                        Real db_hi = 0.0_rt;
                        if (!(split_pec && g == iw))
                        {
                            if (g == 0)
                            {
                                db_lo = db_arr(hi - 1, j, k);
                            }
                            else
                            {
                                db_lo = db_arr(i - 1, j, k);
                            }
                            db_hi = db_arr(i, j, k);
                        }
                        if (split_pec && g == iw - 1)
                        {
                            db_hi = 0.0_rt;
                        }
                        if (split_pec && g == iw)
                        {
                            db_lo = 0.0_rt;
                        }
                        if (split_pec && g == iw + 1)
                        {
                            db_lo = 0.0_rt;
                        }

                        Real const al = db_lo * inv_d2;
                        Real const ga = db_hi * inv_d2;
                        bb[p] = 1.0_rt / cb_arr(i, j, k) + al + ga;
                        a[p] = (p == 0) ? 0.0_rt : -al;
                        c[p] = (p == nsolve - 1) ? 0.0_rt : -ga;
                    }

                    Real gamma = -bb[0];
                    bb[0] -= gamma;
                    bb[nsolve - 1] -= alpha_cyclic * beta_cyclic / gamma;

                    for (int p = 0; p < nsolve; ++p)
                    {
                        int const g = periodicLineGlobalIdx(p, iw);
                        x[p] = field_arr(lo + g, j, k);
                    }

                    solveCyclicTridiagonal(a, bb, c, alpha_cyclic, beta_cyclic, gamma, x, x,
                                           cprime, dprime, z, nsolve);

                    for (int p = 0; p < nsolve; ++p)
                    {
                        int const g = periodicLineGlobalIdx(p, iw);
                        field_arr(lo + g, j, k) = x[p];
                    }
                    if (split_pec)
                    {
                        field_arr(interior_pec_iw, j, k) = 0.0_rt;
                    }
                    field_arr(hi, j, k) = x[0]; });
            }
            else if (solve_dir == 1)
            {
                Box const b2d = amrex::makeSlab(bx, 1, lo);
                Long const nlines = b2d.numPts();
                Gpu::DeviceVector<Real> line_work(nlines * nsolve * n_line_work);
                Gpu::DeviceVector<Real> line_coeff(nlines * nsolve * n_line_coeff);
                Real *work = line_work.data();
                Real *coeff = line_coeff.data();
                int const ilo = b2d.smallEnd(0);
                int const klo = b2d.smallEnd(2);
                int const ilen = b2d.length(0);

                amrex::ParallelForOMP(b2d, [=] AMREX_GPU_DEVICE(int i, int, int k) noexcept
                                      {
                    if (skip_pec_lines &&
                        PecOnPlane(pec_normal, pec_location, pec_lo, pec_hi, i, 0, k))
                    {
                        for (int jj = lo; jj <= hi; ++jj)
                        {
                            field_arr(i, jj, k) = 0.0_rt;
                        }
                        return;
                    }

                    int const line_id = (i - ilo) + (k - klo) * ilen;
                    Real *cprime = work + line_id * nsolve * n_line_work;
                    Real *dprime = cprime + nsolve;
                    Real *x = dprime + nsolve;
                    Real *z = x + nsolve;
                    Real *a = coeff + line_id * nsolve * n_line_coeff;
                    Real *bb = a + nsolve;
                    Real *c = bb + nsolve;

                    Real db_seam = db_arr(i, hi - 1, k);
                    Real alpha_cyclic = -db_seam * inv_d2;
                    Real beta_cyclic = -db_seam * inv_d2;

                    for (int p = 0; p < nsolve; ++p)
                    {
                        int const g = periodicLineGlobalIdx(p, iw);
                        int const j = lo + g;

                        Real db_lo = 0.0_rt;
                        Real db_hi = 0.0_rt;
                        if (!(split_pec && g == iw))
                        {
                            if (g == 0)
                            {
                                db_lo = db_arr(i, hi - 1, k);
                            }
                            else
                            {
                                db_lo = db_arr(i, j - 1, k);
                            }
                            db_hi = db_arr(i, j, k);
                        }
                        if (split_pec && g == iw - 1)
                        {
                            db_hi = 0.0_rt;
                        }
                        if (split_pec && g == iw)
                        {
                            db_lo = 0.0_rt;
                        }
                        if (split_pec && g == iw + 1)
                        {
                            db_lo = 0.0_rt;
                        }

                        Real const al = db_lo * inv_d2;
                        Real const ga = db_hi * inv_d2;
                        bb[p] = 1.0_rt / cb_arr(i, j, k) + al + ga;
                        a[p] = (p == 0) ? 0.0_rt : -al;
                        c[p] = (p == nsolve - 1) ? 0.0_rt : -ga;
                    }

                    Real gamma = -bb[0];
                    bb[0] -= gamma;
                    bb[nsolve - 1] -= alpha_cyclic * beta_cyclic / gamma;

                    for (int p = 0; p < nsolve; ++p)
                    {
                        int const g = periodicLineGlobalIdx(p, iw);
                        x[p] = field_arr(i, lo + g, k);
                    }

                    solveCyclicTridiagonal(a, bb, c, alpha_cyclic, beta_cyclic, gamma, x, x,
                                           cprime, dprime, z, nsolve);

                    for (int p = 0; p < nsolve; ++p)
                    {
                        int const g = periodicLineGlobalIdx(p, iw);
                        field_arr(i, lo + g, k) = x[p];
                    }
                    if (split_pec)
                    {
                        field_arr(i, interior_pec_iw, k) = 0.0_rt;
                    }
                    field_arr(i, hi, k) = x[0]; });
            }
            else if (solve_dir == 2)
            {
                Box const b2d = amrex::makeSlab(bx, 2, lo);
                Long const nlines = b2d.numPts();
                Gpu::DeviceVector<Real> line_work(nlines * nsolve * n_line_work);
                Gpu::DeviceVector<Real> line_coeff(nlines * nsolve * n_line_coeff);
                Real *work = line_work.data();
                Real *coeff = line_coeff.data();
                int const ilo = b2d.smallEnd(0);
                int const jlo = b2d.smallEnd(1);
                int const ilen = b2d.length(0);

                amrex::ParallelForOMP(b2d, [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept
                                      {
                    if (skip_pec_lines &&
                        PecOnPlane(pec_normal, pec_location, pec_lo, pec_hi, i, j, 0))
                    {
                        for (int kk = lo; kk <= hi; ++kk)
                        {
                            field_arr(i, j, kk) = 0.0_rt;
                        }
                        return;
                    }

                    int const line_id = (i - ilo) + (j - jlo) * ilen;
                    Real *cprime = work + line_id * nsolve * n_line_work;
                    Real *dprime = cprime + nsolve;
                    Real *x = dprime + nsolve;
                    Real *z = x + nsolve;
                    Real *a = coeff + line_id * nsolve * n_line_coeff;
                    Real *bb = a + nsolve;
                    Real *c = bb + nsolve;

                    Real db_seam = db_arr(i, j, hi - 1);
                    Real alpha_cyclic = -db_seam * inv_d2;
                    Real beta_cyclic = -db_seam * inv_d2;

                    for (int p = 0; p < nsolve; ++p)
                    {
                        int const g = periodicLineGlobalIdx(p, iw);
                        int const kk = lo + g;

                        Real db_lo = 0.0_rt;
                        Real db_hi = 0.0_rt;
                        if (!(split_pec && g == iw))
                        {
                            if (g == 0)
                            {
                                db_lo = db_arr(i, j, hi - 1);
                            }
                            else
                            {
                                db_lo = db_arr(i, j, kk - 1);
                            }
                            db_hi = db_arr(i, j, kk);
                        }
                        if (split_pec && g == iw - 1)
                        {
                            db_hi = 0.0_rt;
                        }
                        if (split_pec && g == iw)
                        {
                            db_lo = 0.0_rt;
                        }
                        if (split_pec && g == iw + 1)
                        {
                            db_lo = 0.0_rt;
                        }

                        Real const al = db_lo * inv_d2;
                        Real const ga = db_hi * inv_d2;
                        bb[p] = 1.0_rt / cb_arr(i, j, kk) + al + ga;
                        a[p] = (p == 0) ? 0.0_rt : -al;
                        c[p] = (p == nsolve - 1) ? 0.0_rt : -ga;
                    }

                    Real gamma = -bb[0];
                    bb[0] -= gamma;
                    bb[nsolve - 1] -= alpha_cyclic * beta_cyclic / gamma;

                    for (int p = 0; p < nsolve; ++p)
                    {
                        int const g = periodicLineGlobalIdx(p, iw);
                        x[p] = field_arr(i, j, lo + g);
                    }

                    solveCyclicTridiagonal(a, bb, c, alpha_cyclic, beta_cyclic, gamma, x, x,
                                           cprime, dprime, z, nsolve);

                    for (int p = 0; p < nsolve; ++p)
                    {
                        int const g = periodicLineGlobalIdx(p, iw);
                        field_arr(i, j, lo + g) = x[p];
                    }
                    if (split_pec)
                    {
                        field_arr(i, j, interior_pec_iw) = 0.0_rt;
                    }
                    field_arr(i, j, hi) = x[0]; });
            }
            else
            {
                amrex::Abort("solve_dir must be 0, 1, or 2");
            }
        }
    }

    void solveDirichletNodalLines(MultiFab &field,
                                  MultiFab const &rhs,
                                  MultiFab const &Cb,
                                  MultiFab const &Db,
                                  int solve_dir,
                                  Real inv_d2,
                                  std::string const &solver_name)
    {
        Box const domain = field.boxArray().minimalBox();
        int const lo = domain.smallEnd(solve_dir);
        int const hi = domain.bigEnd(solve_dir);
        int const nsolve = hi - lo - 1; // interior unknowns with E(lo)=E(hi)=0

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            nsolve >= 1,
            (solver_name + " requires at least two cells along the implicit direction for PEC")
                .c_str());

        field.ParallelCopy(rhs, 0, 0, 1);

        constexpr int n_line_work = 3; // cprime, dprime, x
        constexpr int n_line_coeff = 3; // a, b, c per row

        for (MFIter mfi(field); mfi.isValid(); ++mfi)
        {
            Box const &bx = mfi.validbox();
            amrex::ignore_unused(solver_name);
            auto const &field_arr = field.array(mfi);
            auto const &cb_arr = Cb.const_array(mfi);
            auto const &db_arr = Db.const_array(mfi);

            if (solve_dir == 0)
            {
                Box const b2d = amrex::makeSlab(bx, 0, lo + 1);
                Long const nlines = b2d.numPts();
                Gpu::DeviceVector<Real> line_work(nlines * nsolve * n_line_work);
                Gpu::DeviceVector<Real> line_coeff(nlines * nsolve * n_line_coeff);
                Real *work = line_work.data();
                Real *coeff = line_coeff.data();
                int const jlo = b2d.smallEnd(1);
                int const klo = b2d.smallEnd(2);
                int const jlen = b2d.length(1);

                amrex::ParallelForOMP(b2d, [=] AMREX_GPU_DEVICE(int, int j, int k) noexcept
                                      {
                    int const line_id = (j - jlo) + (k - klo) * jlen;
                    Real *cprime = work + line_id * nsolve * n_line_work;
                    Real *dprime = cprime + nsolve;
                    Real *x = dprime + nsolve;
                    Real *a = coeff + line_id * nsolve * n_line_coeff;
                    Real *b = a + nsolve;
                    Real *c = b + nsolve;

                    for (int ii = 0; ii < nsolve; ++ii)
                    {
                        int const i = lo + 1 + ii;
                        Real const db_lo = (ii == 0) ? 0.0_rt : db_arr(i - 1, j, k);
                        Real const db_hi = (ii == nsolve - 1) ? 0.0_rt : db_arr(i, j, k);
                        Real const al = db_lo * inv_d2;
                        Real const ga = db_hi * inv_d2;
                        b[ii] = 1.0_rt / cb_arr(i, j, k) + al + ga;
                        a[ii] = (ii == 0) ? 0.0_rt : -al;
                        c[ii] = (ii == nsolve - 1) ? 0.0_rt : -ga;
                        x[ii] = field_arr(i, j, k);
                    }

                    solveTridiagonal(a, b, c, x, x, cprime, dprime, nsolve);

                    for (int ii = 0; ii < nsolve; ++ii)
                    {
                        field_arr(lo + 1 + ii, j, k) = x[ii];
                    }
                    field_arr(lo, j, k) = 0.0_rt;
                    field_arr(hi, j, k) = 0.0_rt; });
            }
            else if (solve_dir == 1)
            {
                Box const b2d = amrex::makeSlab(bx, 1, lo + 1);
                Long const nlines = b2d.numPts();
                Gpu::DeviceVector<Real> line_work(nlines * nsolve * n_line_work);
                Gpu::DeviceVector<Real> line_coeff(nlines * nsolve * n_line_coeff);
                Real *work = line_work.data();
                Real *coeff = line_coeff.data();
                int const ilo = b2d.smallEnd(0);
                int const klo = b2d.smallEnd(2);
                int const ilen = b2d.length(0);

                amrex::ParallelForOMP(b2d, [=] AMREX_GPU_DEVICE(int i, int, int k) noexcept
                                      {
                    int const line_id = (i - ilo) + (k - klo) * ilen;
                    Real *cprime = work + line_id * nsolve * n_line_work;
                    Real *dprime = cprime + nsolve;
                    Real *x = dprime + nsolve;
                    Real *a = coeff + line_id * nsolve * n_line_coeff;
                    Real *b = a + nsolve;
                    Real *c = b + nsolve;

                    for (int jj = 0; jj < nsolve; ++jj)
                    {
                        int const j = lo + 1 + jj;
                        Real const db_lo = (jj == 0) ? 0.0_rt : db_arr(i, j - 1, k);
                        Real const db_hi = (jj == nsolve - 1) ? 0.0_rt : db_arr(i, j, k);
                        Real const al = db_lo * inv_d2;
                        Real const ga = db_hi * inv_d2;
                        b[jj] = 1.0_rt / cb_arr(i, j, k) + al + ga;
                        a[jj] = (jj == 0) ? 0.0_rt : -al;
                        c[jj] = (jj == nsolve - 1) ? 0.0_rt : -ga;
                        x[jj] = field_arr(i, j, k);
                    }

                    solveTridiagonal(a, b, c, x, x, cprime, dprime, nsolve);

                    for (int jj = 0; jj < nsolve; ++jj)
                    {
                        field_arr(i, lo + 1 + jj, k) = x[jj];
                    }
                    field_arr(i, lo, k) = 0.0_rt;
                    field_arr(i, hi, k) = 0.0_rt; });
            }
            else if (solve_dir == 2)
            {
                Box const b2d = amrex::makeSlab(bx, 2, lo + 1);
                Long const nlines = b2d.numPts();
                Gpu::DeviceVector<Real> line_work(nlines * nsolve * n_line_work);
                Gpu::DeviceVector<Real> line_coeff(nlines * nsolve * n_line_coeff);
                Real *work = line_work.data();
                Real *coeff = line_coeff.data();
                int const ilo = b2d.smallEnd(0);
                int const jlo = b2d.smallEnd(1);
                int const ilen = b2d.length(0);

                amrex::ParallelForOMP(b2d, [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept
                                      {
                    int const line_id = (i - ilo) + (j - jlo) * ilen;
                    Real *cprime = work + line_id * nsolve * n_line_work;
                    Real *dprime = cprime + nsolve;
                    Real *x = dprime + nsolve;
                    Real *a = coeff + line_id * nsolve * n_line_coeff;
                    Real *b = a + nsolve;
                    Real *c = b + nsolve;

                    for (int kk = 0; kk < nsolve; ++kk)
                    {
                        int const k = lo + 1 + kk;
                        Real const db_lo = (kk == 0) ? 0.0_rt : db_arr(i, j, k - 1);
                        Real const db_hi = (kk == nsolve - 1) ? 0.0_rt : db_arr(i, j, k);
                        Real const al = db_lo * inv_d2;
                        Real const ga = db_hi * inv_d2;
                        b[kk] = 1.0_rt / cb_arr(i, j, k) + al + ga;
                        a[kk] = (kk == 0) ? 0.0_rt : -al;
                        c[kk] = (kk == nsolve - 1) ? 0.0_rt : -ga;
                        x[kk] = field_arr(i, j, k);
                    }

                    solveTridiagonal(a, b, c, x, x, cprime, dprime, nsolve);

                    for (int kk = 0; kk < nsolve; ++kk)
                    {
                        field_arr(i, j, lo + 1 + kk) = x[kk];
                    }
                    field_arr(i, j, lo) = 0.0_rt;
                    field_arr(i, j, hi) = 0.0_rt; });
            }
            else
            {
                amrex::Abort("solve_dir must be 0, 1, or 2");
            }
        }
    }

    void copyCoefToLayout(MultiFab &dst, MultiFab const &src, Periodicity const &period)
    {
        dst.ParallelCopy(src, 0, 0, 1, IntVect(0), dst.nGrowVect(), period);
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
    pp.query("dt", m_dt);
    pp.query("output_dir", m_output_dir);
    pp.query("conv_plt", m_conv_plt);

    const bool use_dt = (m_dt > 0 && m_cfl < 0);
    const bool use_cfl = (m_dt < 0 && m_cfl > 0);
    if (!use_dt && !use_cfl)
    {
        if (m_dt > 0 && m_cfl > 0)
        {
            amrex::Abort("adi.dt and adi.cfl are both positive; set one < 0 to disable it");
        }
        if (m_dt < 0 && m_cfl < 0)
        {
            amrex::Abort("adi.dt and adi.cfl are both negative; set one > 0");
        }
        amrex::Abort("adi.dt and adi.cfl: exactly one must be > 0 and the other < 0");
    }

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

    m_pec_normal = -1;
    pp.query("pec_normal", m_pec_normal);
    if (m_pec_normal < -1 || m_pec_normal > 2)
    {
        amrex::Abort("adi.pec_normal must be -1 (none) or 0, 1, 2");
    }

    m_pec_location = 0;
    pp.query("pec_location", m_pec_location);
    if (m_pec_location != 0 && m_pec_normal < 0)
    {
        amrex::Abort("adi.pec_location requires adi.pec_normal >= 0");
    }

    Box domain(IntVect(0), m_n_cells - 1);
    if (m_pec_location != 0)
    {
        int const pec_lo = domain.smallEnd(m_pec_normal);
        int const pec_hi = domain.bigEnd(m_pec_normal);
        if (m_pec_location <= pec_lo || m_pec_location >= pec_hi)
        {
            amrex::Abort("adi.pec_location must be strictly interior along adi.pec_normal");
        }
    }

    RealBox real_box(prob_lo.begin(), prob_hi.begin());
    Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1, 1, 1)};
    if (m_pec_normal >= 0 && m_pec_location == 0)
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

    ParmParse pp_mat("adi");
    pp_mat.query("eps_r", m_eps_r);
    pp_mat.query("mu_r", m_mu_r);
    pp_mat.query("sigma", m_sigma_bg);

    RealVect block_lo, block_hi;
    if (pp_mat.queryarr("material_block_lo", block_lo) &&
        pp_mat.queryarr("material_block_hi", block_hi))
    {
        m_block_lo = block_lo;
        m_block_hi = block_hi;
        m_has_material_block = true;
        pp_mat.query("block_eps_r", m_block_eps_r);
        pp_mat.query("block_mu_r", m_block_mu_r);
        pp_mat.query("block_sigma", m_block_sigma);
    }

    initMaterials();
}

void ADI::initMaterials()
{
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        IntVect etyp(1);
        etyp[idim] = 0;
        m_Cb[idim].define(amrex::convert(m_grids, etyp), m_dmap, 1, 1);
        m_p[idim].define(amrex::convert(m_grids, etyp), m_dmap, 1, 1);

        IntVect btyp(0);
        btyp[idim] = 1;
        m_Db[idim].define(amrex::convert(m_grids, btyp), m_dmap, 1, 1);
    }
}

void ADI::updateMaterialCoeffs(Real dt)
{
    auto problo = m_geom.ProbLoArray();
    auto dx = m_geom.CellSizeArray();
    Real const eps_bg = eps0 * m_eps_r;
    Real const mu_bg = mu0 * m_mu_r;
    Real const block_eps = eps0 * m_block_eps_r;
    Real const block_mu = mu0 * m_block_mu_r;
    bool const has_block = m_has_material_block;
    GpuArray<Real, AMREX_SPACEDIM> const block_lo{
        AMREX_D_DECL(m_block_lo[0], m_block_lo[1], m_block_lo[2])};
    GpuArray<Real, AMREX_SPACEDIM> const block_hi{
        AMREX_D_DECL(m_block_hi[0], m_block_hi[1], m_block_hi[2])};
    Real const sigma_bg = m_sigma_bg;
    Real const block_sigma = m_block_sigma;

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        auto const &cba = m_Cb[idim].arrays();
        auto const &pa = m_p[idim].arrays();
        bool const e_cc_x = (idim == 0);
        bool const e_cc_y = (idim == 1);
        bool const e_cc_z = (idim == 2);

        ParallelFor(m_Cb[idim], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                    {
            Real const x = problo[0] + (i + (e_cc_x ? 0.5_rt : 0.0_rt)) * dx[0];
            Real const y = problo[1] + (j + (e_cc_y ? 0.5_rt : 0.0_rt)) * dx[1];
            Real const z = problo[2] + (k + (e_cc_z ? 0.5_rt : 0.0_rt)) * dx[2];
            bool const in_block = has_block &&
                                  x >= block_lo[0] && x <= block_hi[0] &&
                                  y >= block_lo[1] && y <= block_hi[1] &&
                                  z >= block_lo[2] && z <= block_hi[2];
            Real const eps = in_block ? block_eps : eps_bg;
            Real const sigma = in_block ? block_sigma : sigma_bg;
            Real const denom = 4.0_rt * eps + sigma * dt;
            Real const Ca = (4.0_rt * eps - sigma * dt) / denom;
            Real const Cb = 2.0_rt * dt / denom;
            cba[b](i, j, k) = Cb;
            pa[b](i, j, k) = Ca / Cb; });

        auto const &dba = m_Db[idim].arrays();
        bool const h_cc_x = (idim != 0);
        bool const h_cc_y = (idim != 1);
        bool const h_cc_z = (idim != 2);

        ParallelFor(m_Db[idim], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                    {
            Real const x = problo[0] + (i + (h_cc_x ? 0.5_rt : 0.0_rt)) * dx[0];
            Real const y = problo[1] + (j + (h_cc_y ? 0.5_rt : 0.0_rt)) * dx[1];
            Real const z = problo[2] + (k + (h_cc_z ? 0.5_rt : 0.0_rt)) * dx[2];
            bool const in_block = has_block &&
                                  x >= block_lo[0] && x <= block_hi[0] &&
                                  y >= block_lo[1] && y <= block_hi[1] &&
                                  z >= block_lo[2] && z <= block_hi[2];
            Real const mu = in_block ? block_mu : mu_bg;
            dba[b](i, j, k) = dt / (2.0_rt * mu); });
    }

    auto const period = m_geom.periodicity();
    Vector<MultiFab *> cb_ptrs{AMREX_D_DECL(&m_Cb[0], &m_Cb[1], &m_Cb[2])};
    Vector<MultiFab *> p_ptrs{AMREX_D_DECL(&m_p[0], &m_p[1], &m_p[2])};
    Vector<MultiFab *> db_ptrs{AMREX_D_DECL(&m_Db[0], &m_Db[1], &m_Db[2])};
    amrex::FillBoundary(cb_ptrs, period);
    amrex::FillBoundary(p_ptrs, period);
    amrex::FillBoundary(db_ptrs, period);
}

void ADI::convertBtoH(Real dt)
{
    if (m_magnetic_fields_are_h)
    {
        return;
    }

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        auto const &dba = m_Db[idim].const_arrays();
        auto const &ha = m_bfields[idim].arrays();
        ParallelFor(m_bfields[idim], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                    { ha[b](i, j, k) *= 2.0_rt * dba[b](i, j, k) / dt; });
    }
    m_magnetic_fields_are_h = true;
}

void ADI::initData()
{
    InitSetupFields("adi", m_ic, m_ic_amplitude, m_ic_dir,
                    m_ic_pol, m_ic_wavelength, m_pulse_center, m_pulse_sigma,
                    m_geom, m_efields, m_bfields);
    m_magnetic_fields_are_h = false;
    PecPinTangentialE(m_pec_normal, m_pec_location, m_efields);
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
    Real dt;
    if (m_dt > 0 && m_cfl < 0)
    {
        dt = m_dt;
    }
    else
    {
        dt = m_cfl / (c * std::sqrt(inv_dx2_sum));
    }

    updateMaterialCoeffs(dt);
    convertBtoH(dt);

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
                            m_conv_plt, m_ic_dir,
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
                                m_conv_plt, m_ic_dir,
                                m_grids, m_dmap, m_geom, m_efields, m_bfields);
        }
    }
}

void ADI::adiFirstHalfStep(Array<MultiFab, AMREX_SPACEDIM> &efields,
                           Array<MultiFab, AMREX_SPACEDIM> &bfields,
                           FieldArray &efields_x, FieldArray &efields_y,
                           FieldArray &efields_z, FieldArray &bfields_x,
                           FieldArray &bfields_y, FieldArray &bfields_z,
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

    PecPinTangentialE(m_pec_normal, m_pec_location, efields);

    stepBx(bfields[0], efields[1], eold[2], dt);
    stepBy(bfields[1], efields[2], eold[0], dt);
    stepBz(bfields[2], efields[0], eold[1], dt);

    amrex::FillBoundary(bfield_ptrs, period);
}

void ADI::adiSecondHalfStep(Array<MultiFab, AMREX_SPACEDIM> &efields,
                            Array<MultiFab, AMREX_SPACEDIM> &bfields,
                            FieldArray &efields_x, FieldArray &efields_y,
                            FieldArray &efields_z, FieldArray &bfields_x,
                            FieldArray &bfields_y, FieldArray &bfields_z,
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

    PecPinTangentialE(m_pec_normal, m_pec_location, efields);

    stepBx(bfields[0], eold[1], efields[2], dt);
    stepBy(bfields[1], eold[2], efields[0], dt);
    stepBz(bfields[2], eold[0], efields[1], dt);

    amrex::FillBoundary(bfield_ptrs, period);
}

MultiFab ADI::buildRhsEx1(Array<MultiFab, AMREX_SPACEDIM> const &efields,
                          Array<MultiFab, AMREX_SPACEDIM> const &bfields,
                          Real dt) const
{
    amrex::ignore_unused(dt);
    // eq:first-e-ex-adi-tridiagonal — implicit along y.
    MultiFab rhs = makeRhsLike(efields[0]);
    MultiFab p_field = makeRhsLike(efields[0]);
    MultiFab db_field = makeRhsLike(efields[0]);
    auto const period = m_geom.periodicity();
    copyCoefToLayout(p_field, m_p[0], period);
    copyCoefToLayout(db_field, m_Db[2], period);

    auto const dx = m_geom.CellSizeArray();
    Real const dyinv = 1.0_rt / dx[1];
    Real const dzinv = 1.0_rt / dx[2];
    Real const dxinv = 1.0_rt / dx[0];

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box &tilebx = mfi.tilebox();
        auto const &rhs_arr = rhs.array(mfi);
        auto const &ex_arr = efields[0].const_array(mfi);
        auto const &p_arr = p_field.const_array(mfi);
        auto const &db_arr = db_field.const_array(mfi);
        auto const &ey_arr = efields[1].const_array(mfi);
        auto const &hz_arr = bfields[2].const_array(mfi);
        auto const &hy_arr = bfields[1].const_array(mfi);

        ParallelFor(tilebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
            Real const q = db_arr(i, j - 1, k);
            Real const r = db_arr(i, j, k);
            Real const curl_h = dyinv * (hz_arr(i, j, k) - hz_arr(i, j - 1, k)) -
                                dzinv * (hy_arr(i, j, k) - hy_arr(i, j, k - 1));
            Real const ey_lo = ey_arr(i + 1, j - 1, k) - ey_arr(i, j - 1, k);
            Real const ey_hi = ey_arr(i + 1, j, k) - ey_arr(i, j, k);
            rhs_arr(i, j, k) = p_arr(i, j, k) * ex_arr(i, j, k) + curl_h +
                               q * dxinv * dyinv * ey_lo - r * dxinv * dyinv * ey_hi; });
    }

    return rhs;
}

MultiFab ADI::buildRhsEy1(Array<MultiFab, AMREX_SPACEDIM> const &efields,
                          Array<MultiFab, AMREX_SPACEDIM> const &bfields,
                          Real dt) const
{
    amrex::ignore_unused(dt);
    // eq:first-e-ey-adi-tridiagonal — implicit along z.
    MultiFab rhs = makeRhsLike(efields[1]);
    MultiFab p_field = makeRhsLike(efields[1]);
    MultiFab db_field = makeRhsLike(efields[1]);
    auto const period = m_geom.periodicity();
    copyCoefToLayout(p_field, m_p[1], period);
    copyCoefToLayout(db_field, m_Db[0], period);

    auto const dx = m_geom.CellSizeArray();
    Real const dzinv = 1.0_rt / dx[2];
    Real const dxinv = 1.0_rt / dx[0];
    Real const dyinv = 1.0_rt / dx[1];

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box &tilebx = mfi.tilebox();
        auto const &rhs_arr = rhs.array(mfi);
        auto const &ey_arr = efields[1].const_array(mfi);
        auto const &p_arr = p_field.const_array(mfi);
        auto const &db_arr = db_field.const_array(mfi);
        auto const &ez_arr = efields[2].const_array(mfi);
        auto const &hx_arr = bfields[0].const_array(mfi);
        auto const &hz_arr = bfields[2].const_array(mfi);

        ParallelFor(tilebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
            Real const q = db_arr(i, j, k - 1);
            Real const r = db_arr(i, j, k);
            Real const curl_h = dzinv * (hx_arr(i, j, k) - hx_arr(i, j, k - 1)) -
                                dxinv * (hz_arr(i, j, k) - hz_arr(i - 1, j, k));
            Real const ez_lo = ez_arr(i, j + 1, k - 1) - ez_arr(i, j, k - 1);
            Real const ez_hi = ez_arr(i, j + 1, k) - ez_arr(i, j, k);
            rhs_arr(i, j, k) = p_arr(i, j, k) * ey_arr(i, j, k) + curl_h +
                               q * dyinv * dzinv * ez_lo - r * dyinv * dzinv * ez_hi; });
    }

    return rhs;
}

MultiFab ADI::buildRhsEz1(Array<MultiFab, AMREX_SPACEDIM> const &efields,
                          Array<MultiFab, AMREX_SPACEDIM> const &bfields,
                          Real dt) const
{
    amrex::ignore_unused(dt);
    // eq:first-e-ez-adi-tridiagonal — implicit along x.
    MultiFab rhs = makeRhsLike(efields[2]);
    MultiFab p_field = makeRhsLike(efields[2]);
    MultiFab db_field = makeRhsLike(efields[2]);
    auto const period = m_geom.periodicity();
    copyCoefToLayout(p_field, m_p[2], period);
    copyCoefToLayout(db_field, m_Db[1], period);

    auto const dx = m_geom.CellSizeArray();
    Real const dxinv = 1.0_rt / dx[0];
    Real const dyinv = 1.0_rt / dx[1];
    Real const dzinv = 1.0_rt / dx[2];

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box &tilebx = mfi.tilebox();
        auto const &rhs_arr = rhs.array(mfi);
        auto const &ez_arr = efields[2].const_array(mfi);
        auto const &p_arr = p_field.const_array(mfi);
        auto const &db_arr = db_field.const_array(mfi);
        auto const &ex_arr = efields[0].const_array(mfi);
        auto const &hy_arr = bfields[1].const_array(mfi);
        auto const &hx_arr = bfields[0].const_array(mfi);

        ParallelFor(tilebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
            Real const q = db_arr(i - 1, j, k);
            Real const r = db_arr(i, j, k);
            Real const curl_h = dxinv * (hy_arr(i, j, k) - hy_arr(i - 1, j, k)) -
                                dyinv * (hx_arr(i, j, k) - hx_arr(i, j - 1, k));
            Real const ex_lo = ex_arr(i - 1, j, k + 1) - ex_arr(i - 1, j, k);
            Real const ex_hi = ex_arr(i, j, k + 1) - ex_arr(i, j, k);
            rhs_arr(i, j, k) = p_arr(i, j, k) * ez_arr(i, j, k) + curl_h +
                               q * dzinv * dxinv * ex_lo - r * dzinv * dxinv * ex_hi; });
    }

    return rhs;
}

void ADI::solveImplicitEx1(MultiFab &ex, MultiFab const &rhs, Real dt) const
{
    amrex::ignore_unused(dt);
    MultiFab Cb = makeRhsLike(ex);
    MultiFab Db = makeRhsLike(ex);
    auto const period = m_geom.periodicity();
    copyCoefToLayout(Cb, m_Cb[0], period);
    copyCoefToLayout(Db, m_Db[2], period);
    Real const dy = m_geom.CellSizeArray()[1];
    Real const inv_d2 = 1.0_rt / (dy * dy);
    if (m_pec_location == 0 && m_pec_normal == 1)
    {
        solveDirichletNodalLines(ex, rhs, Cb, Db, 1, inv_d2, "solveImplicitEx1");
    }
    else if (m_pec_location != 0 && m_pec_normal == 1)
    {
        solvePeriodicCyclicLines(ex, rhs, Cb, Db, 1, inv_d2, m_pec_location, -1, 0, -1,
                                 "solveImplicitEx1");
    }
    else
    {
        solvePeriodicCyclicLines(ex, rhs, Cb, Db, 1, inv_d2, -1, m_pec_normal, m_pec_location, 0,
                                 "solveImplicitEx1");
    }
}

void ADI::solveImplicitEy1(MultiFab &ey, MultiFab const &rhs, Real dt) const
{
    amrex::ignore_unused(dt);
    MultiFab Cb = makeRhsLike(ey);
    MultiFab Db = makeRhsLike(ey);
    auto const period = m_geom.periodicity();
    copyCoefToLayout(Cb, m_Cb[1], period);
    copyCoefToLayout(Db, m_Db[0], period);
    Real const dz = m_geom.CellSizeArray()[2];
    Real const inv_d2 = 1.0_rt / (dz * dz);
    if (m_pec_location == 0 && m_pec_normal == 2)
    {
        solveDirichletNodalLines(ey, rhs, Cb, Db, 2, inv_d2, "solveImplicitEy1");
    }
    else if (m_pec_location != 0 && m_pec_normal == 2)
    {
        solvePeriodicCyclicLines(ey, rhs, Cb, Db, 2, inv_d2, m_pec_location, -1, 0, -1,
                                 "solveImplicitEy1");
    }
    else
    {
        solvePeriodicCyclicLines(ey, rhs, Cb, Db, 2, inv_d2, -1, m_pec_normal, m_pec_location, 1,
                                 "solveImplicitEy1");
    }
}

void ADI::solveImplicitEz1(MultiFab &ez, MultiFab const &rhs, Real dt) const
{
    amrex::ignore_unused(dt);
    MultiFab Cb = makeRhsLike(ez);
    MultiFab Db = makeRhsLike(ez);
    auto const period = m_geom.periodicity();
    copyCoefToLayout(Cb, m_Cb[2], period);
    copyCoefToLayout(Db, m_Db[1], period);
    Real const dx = m_geom.CellSizeArray()[0];
    Real const inv_d2 = 1.0_rt / (dx * dx);
    if (m_pec_location == 0 && m_pec_normal == 0)
    {
        solveDirichletNodalLines(ez, rhs, Cb, Db, 0, inv_d2, "solveImplicitEz1");
    }
    else if (m_pec_location != 0 && m_pec_normal == 0)
    {
        solvePeriodicCyclicLines(ez, rhs, Cb, Db, 0, inv_d2, m_pec_location, -1, 0, -1,
                                 "solveImplicitEz1");
    }
    else
    {
        solvePeriodicCyclicLines(ez, rhs, Cb, Db, 0, inv_d2, -1, m_pec_normal, m_pec_location, 2,
                                 "solveImplicitEz1");
    }
}

MultiFab ADI::buildRhsEx2(Array<MultiFab, AMREX_SPACEDIM> const &efields,
                          Array<MultiFab, AMREX_SPACEDIM> const &bfields,
                          Real dt) const
{
    amrex::ignore_unused(dt);
    // eq:second-e-ex-adi-tridiagonal — implicit along z.
    MultiFab rhs = makeRhsLike(efields[0]);
    MultiFab p_field = makeRhsLike(efields[0]);
    MultiFab db_field = makeRhsLike(efields[0]);
    auto const period = m_geom.periodicity();
    copyCoefToLayout(p_field, m_p[0], period);
    copyCoefToLayout(db_field, m_Db[1], period);

    auto const dx = m_geom.CellSizeArray();
    Real const dyinv = 1.0_rt / dx[1];
    Real const dzinv = 1.0_rt / dx[2];
    Real const dxinv = 1.0_rt / dx[0];

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box &tilebx = mfi.tilebox();
        auto const &rhs_arr = rhs.array(mfi);
        auto const &ex_arr = efields[0].const_array(mfi);
        auto const &p_arr = p_field.const_array(mfi);
        auto const &db_arr = db_field.const_array(mfi);
        auto const &ez_arr = efields[2].const_array(mfi);
        auto const &hz_arr = bfields[2].const_array(mfi);
        auto const &hy_arr = bfields[1].const_array(mfi);

        ParallelFor(tilebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
            Real const q = db_arr(i, j, k - 1);
            Real const r = db_arr(i, j, k);
            Real const curl_h = dyinv * (hz_arr(i, j, k) - hz_arr(i, j - 1, k)) -
                                dzinv * (hy_arr(i, j, k) - hy_arr(i, j, k - 1));
            Real const ez_lo = ez_arr(i + 1, j, k - 1) - ez_arr(i, j, k - 1);
            Real const ez_hi = ez_arr(i + 1, j, k) - ez_arr(i, j, k);
            rhs_arr(i, j, k) = p_arr(i, j, k) * ex_arr(i, j, k) + curl_h +
                               q * dxinv * dzinv * ez_lo - r * dxinv * dzinv * ez_hi; });
    }

    return rhs;
}

MultiFab ADI::buildRhsEy2(Array<MultiFab, AMREX_SPACEDIM> const &efields,
                          Array<MultiFab, AMREX_SPACEDIM> const &bfields,
                          Real dt) const
{
    amrex::ignore_unused(dt);
    // eq:second-e-ey-adi-tridiagonal — implicit along x.
    MultiFab rhs = makeRhsLike(efields[1]);
    MultiFab p_field = makeRhsLike(efields[1]);
    MultiFab db_field = makeRhsLike(efields[1]);
    auto const period = m_geom.periodicity();
    copyCoefToLayout(p_field, m_p[1], period);
    copyCoefToLayout(db_field, m_Db[2], period);

    auto const dx = m_geom.CellSizeArray();
    Real const dzinv = 1.0_rt / dx[2];
    Real const dxinv = 1.0_rt / dx[0];
    Real const dyinv = 1.0_rt / dx[1];

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box &tilebx = mfi.tilebox();
        auto const &rhs_arr = rhs.array(mfi);
        auto const &ey_arr = efields[1].const_array(mfi);
        auto const &p_arr = p_field.const_array(mfi);
        auto const &db_arr = db_field.const_array(mfi);
        auto const &ex_arr = efields[0].const_array(mfi);
        auto const &hx_arr = bfields[0].const_array(mfi);
        auto const &hz_arr = bfields[2].const_array(mfi);

        ParallelFor(tilebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
            Real const q = db_arr(i - 1, j, k);
            Real const r = db_arr(i, j, k);
            Real const curl_h = dzinv * (hx_arr(i, j, k) - hx_arr(i, j, k - 1)) -
                                dxinv * (hz_arr(i, j, k) - hz_arr(i - 1, j, k));
            Real const ex_lo = ex_arr(i - 1, j + 1, k) - ex_arr(i - 1, j, k);
            Real const ex_hi = ex_arr(i, j + 1, k) - ex_arr(i, j, k);
            rhs_arr(i, j, k) = p_arr(i, j, k) * ey_arr(i, j, k) + curl_h +
                               q * dyinv * dxinv * ex_lo - r * dyinv * dxinv * ex_hi; });
    }

    return rhs;
}

MultiFab ADI::buildRhsEz2(Array<MultiFab, AMREX_SPACEDIM> const &efields,
                          Array<MultiFab, AMREX_SPACEDIM> const &bfields,
                          Real dt) const
{
    amrex::ignore_unused(dt);
    // eq:second-e-ez-adi-tridiagonal — implicit along y.
    MultiFab rhs = makeRhsLike(efields[2]);
    MultiFab p_field = makeRhsLike(efields[2]);
    MultiFab db_field = makeRhsLike(efields[2]);
    auto const period = m_geom.periodicity();
    copyCoefToLayout(p_field, m_p[2], period);
    copyCoefToLayout(db_field, m_Db[0], period);

    auto const dx = m_geom.CellSizeArray();
    Real const dxinv = 1.0_rt / dx[0];
    Real const dyinv = 1.0_rt / dx[1];
    Real const dzinv = 1.0_rt / dx[2];

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box &tilebx = mfi.tilebox();
        auto const &rhs_arr = rhs.array(mfi);
        auto const &ez_arr = efields[2].const_array(mfi);
        auto const &p_arr = p_field.const_array(mfi);
        auto const &db_arr = db_field.const_array(mfi);
        auto const &ey_arr = efields[1].const_array(mfi);
        auto const &hy_arr = bfields[1].const_array(mfi);
        auto const &hx_arr = bfields[0].const_array(mfi);

        ParallelFor(tilebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
            Real const q = db_arr(i, j - 1, k);
            Real const r = db_arr(i, j, k);
            Real const curl_h = dxinv * (hy_arr(i, j, k) - hy_arr(i - 1, j, k)) -
                                dyinv * (hx_arr(i, j, k) - hx_arr(i, j - 1, k));
            Real const ey_lo = ey_arr(i, j - 1, k + 1) - ey_arr(i, j - 1, k);
            Real const ey_hi = ey_arr(i, j, k + 1) - ey_arr(i, j, k);
            rhs_arr(i, j, k) = p_arr(i, j, k) * ez_arr(i, j, k) + curl_h +
                               q * dzinv * dyinv * ey_lo - r * dzinv * dyinv * ey_hi; });
    }

    return rhs;
}

void ADI::solveImplicitEx2(MultiFab &ex, MultiFab const &rhs, Real dt) const
{
    amrex::ignore_unused(dt);
    MultiFab Cb = makeRhsLike(ex);
    MultiFab Db = makeRhsLike(ex);
    auto const period = m_geom.periodicity();
    copyCoefToLayout(Cb, m_Cb[0], period);
    copyCoefToLayout(Db, m_Db[1], period);
    Real const dz = m_geom.CellSizeArray()[2];
    Real const inv_d2 = 1.0_rt / (dz * dz);
    if (m_pec_location == 0 && m_pec_normal == 2)
    {
        solveDirichletNodalLines(ex, rhs, Cb, Db, 2, inv_d2, "solveImplicitEx2");
    }
    else if (m_pec_location != 0 && m_pec_normal == 2)
    {
        solvePeriodicCyclicLines(ex, rhs, Cb, Db, 2, inv_d2, m_pec_location, -1, 0, -1,
                                 "solveImplicitEx2");
    }
    else
    {
        solvePeriodicCyclicLines(ex, rhs, Cb, Db, 2, inv_d2, -1, m_pec_normal, m_pec_location, 0,
                                 "solveImplicitEx2");
    }
}

void ADI::solveImplicitEy2(MultiFab &ey, MultiFab const &rhs, Real dt) const
{
    amrex::ignore_unused(dt);
    MultiFab Cb = makeRhsLike(ey);
    MultiFab Db = makeRhsLike(ey);
    auto const period = m_geom.periodicity();
    copyCoefToLayout(Cb, m_Cb[1], period);
    copyCoefToLayout(Db, m_Db[2], period);
    Real const dx = m_geom.CellSizeArray()[0];
    Real const inv_d2 = 1.0_rt / (dx * dx);
    if (m_pec_location == 0 && m_pec_normal == 0)
    {
        solveDirichletNodalLines(ey, rhs, Cb, Db, 0, inv_d2, "solveImplicitEy2");
    }
    else if (m_pec_location != 0 && m_pec_normal == 0)
    {
        solvePeriodicCyclicLines(ey, rhs, Cb, Db, 0, inv_d2, m_pec_location, -1, 0, -1,
                                 "solveImplicitEy2");
    }
    else
    {
        solvePeriodicCyclicLines(ey, rhs, Cb, Db, 0, inv_d2, -1, m_pec_normal, m_pec_location, 1,
                                 "solveImplicitEy2");
    }
}

void ADI::solveImplicitEz2(MultiFab &ez, MultiFab const &rhs, Real dt) const
{
    amrex::ignore_unused(dt);
    MultiFab Cb = makeRhsLike(ez);
    MultiFab Db = makeRhsLike(ez);
    auto const period = m_geom.periodicity();
    copyCoefToLayout(Cb, m_Cb[2], period);
    copyCoefToLayout(Db, m_Db[0], period);
    Real const dy = m_geom.CellSizeArray()[1];
    Real const inv_d2 = 1.0_rt / (dy * dy);
    if (m_pec_location == 0 && m_pec_normal == 1)
    {
        solveDirichletNodalLines(ez, rhs, Cb, Db, 1, inv_d2, "solveImplicitEz2");
    }
    else if (m_pec_location != 0 && m_pec_normal == 1)
    {
        solvePeriodicCyclicLines(ez, rhs, Cb, Db, 1, inv_d2, m_pec_location, -1, 0, -1,
                                 "solveImplicitEz2");
    }
    else
    {
        solvePeriodicCyclicLines(ez, rhs, Cb, Db, 1, inv_d2, -1, m_pec_normal, m_pec_location, 2,
                                 "solveImplicitEz2");
    }
}

void ADI::stepBx(MultiFab &hx_dst, MultiFab const &ey_src,
                 MultiFab const &ez_src, Real dt)
{
    amrex::ignore_unused(dt);
    // H_x += D_b (dEy/dz - dEz/dy), eq:adi-first-hx / eq:adi-second-hx.
    auto const dxinv = m_geom.InvCellSizeArray();
    auto const &db = m_Db[0].const_arrays();
    auto const &ey = ey_src.const_arrays();
    auto const &ez = ez_src.const_arrays();
    auto const &hx = hx_dst.arrays();

    ParallelFor(hx_dst, [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                {
        Real const r = db[b](i, j, k);
        hx[b](i, j, k) += r * (dxinv[2] * (ey[b](i, j, k + 1) - ey[b](i, j, k)) -
                               dxinv[1] * (ez[b](i, j + 1, k) - ez[b](i, j, k))); });
}

void ADI::stepBy(MultiFab &hy_dst, MultiFab const &ez_src,
                 MultiFab const &ex_src, Real dt)
{
    amrex::ignore_unused(dt);
    auto const dxinv = m_geom.InvCellSizeArray();
    auto const &db = m_Db[1].const_arrays();
    auto const &ex = ex_src.const_arrays();
    auto const &ez = ez_src.const_arrays();
    auto const &hy = hy_dst.arrays();

    ParallelFor(hy_dst, [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                {
        Real const r = db[b](i, j, k);
        hy[b](i, j, k) += r * (dxinv[0] * (ez[b](i + 1, j, k) - ez[b](i, j, k)) -
                               dxinv[2] * (ex[b](i, j, k + 1) - ex[b](i, j, k))); });
}

void ADI::stepBz(MultiFab &hz_dst, MultiFab const &ex_src,
                 MultiFab const &ey_src, Real dt)
{
    amrex::ignore_unused(dt);
    auto const dxinv = m_geom.InvCellSizeArray();
    auto const &db = m_Db[2].const_arrays();
    auto const &ex = ex_src.const_arrays();
    auto const &ey = ey_src.const_arrays();
    auto const &hz = hz_dst.arrays();

    ParallelFor(hz_dst, [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                {
        Real const r = db[b](i, j, k);
        hz[b](i, j, k) += r * (dxinv[1] * (ex[b](i, j + 1, k) - ex[b](i, j, k)) -
                               dxinv[0] * (ey[b](i + 1, j, k) - ey[b](i, j, k))); });
}
