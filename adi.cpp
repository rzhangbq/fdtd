
#include "adi.H"
#include "init.H"
#include "pec.H"
#include "util.H"

#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>

#include <cmath>

using namespace amrex;

namespace
{
    constexpr Real eps0 = 8.854187817e-12;
    constexpr Real mu0 = 4.0 * M_PI * 1e-7;

    MultiFab makeRhsLike(MultiFab const &field)
    {
        return MultiFab(field.boxArray(), field.DistributionMap(), 1, 0);
    }

    MultiFab makeCoeffLike(MultiFab const &field, MultiFab const &coef)
    {
        BoxArray ba(field.boxArray());
        ba.convert(coef.ixType());
        return MultiFab(ba, field.DistributionMap(), 1, coef.nGrowVect());
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

    template <typename Arr>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real lineValue(Arr const &arr,
                                                            int solve_dir,
                                                            int s,
                                                            int i,
                                                            int j,
                                                            int k) noexcept
    {
        if (solve_dir == 0)
        {
            return arr(s, j, k);
        }
        if (solve_dir == 1)
        {
            return arr(i, s, k);
        }
        return arr(i, j, s);
    }

    template <typename Arr>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void setLineValue(Arr const &arr,
                                                               int solve_dir,
                                                               int s,
                                                               int i,
                                                               int j,
                                                               int k,
                                                               Real value) noexcept
    {
        if (solve_dir == 0)
        {
            arr(s, j, k) = value;
        }
        else if (solve_dir == 1)
        {
            arr(i, s, k) = value;
        }
        else
        {
            arr(i, j, s) = value;
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool pecOnSolveSlab(int pec_normal,
                                                                 int pec_location,
                                                                 int pec_lo,
                                                                 int pec_hi,
                                                                 int solve_dir,
                                                                 int i,
                                                                 int j,
                                                                 int k) noexcept
    {
        if (solve_dir == 0)
        {
            return PecOnPlane(pec_normal, pec_location, pec_lo, pec_hi, 0, j, k);
        }
        if (solve_dir == 1)
        {
            return PecOnPlane(pec_normal, pec_location, pec_lo, pec_hi, i, 0, k);
        }
        return PecOnPlane(pec_normal, pec_location, pec_lo, pec_hi, i, j, 0);
    }

    void PinPmlNodalBoundaryPlanes(int pml_normal,
                                   Array<MultiFab, AMREX_SPACEDIM> &fields)
    {
        if (pml_normal < 0)
        {
            return;
        }

        for (int comp = 0; comp < AMREX_SPACEDIM; ++comp)
        {
            MultiFab &field = fields[comp];
            if (field.ixType().cellCentered(pml_normal))
            {
                continue;
            }

            Box const bounds = field.boxArray().minimalBox();
            int const lo = bounds.smallEnd(pml_normal);
            int const hi = bounds.bigEnd(pml_normal);
            auto const &arrs = field.arrays();

            ParallelFor(field, [=] AMREX_GPU_DEVICE(int b, int i, int j, int k) noexcept
                        {
                if (PecOnWallPlane(pml_normal, lo, hi, i, j, k))
                {
                    arrs[b](i, j, k) = 0.0_rt;
                } });
        }
    }

    // Periodic cyclic tridiagonal solve along solve_dir with inhomogeneous coefficients.
    // interior_pec_iw < 0: full nodal line (hi duplicates lo); optional tangential PEC skip via pec_* / e_comp.
    // interior_pec_iw >= 0: pin that interior PEC row with A(iw,iw)=1 and RHS(iw)=0.
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
        int const nsolve = hi - lo;
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

        constexpr int n_line_work = 4;  // cprime, dprime, x, z
        constexpr int n_line_coeff = 3; // a, bb, c per row

        for (MFIter mfi(field); mfi.isValid(); ++mfi)
        {
            Box const &bx = mfi.validbox();
            amrex::ignore_unused(solver_name);
            auto const &field_arr = field.array(mfi);
            auto const &cb_arr = Cb.const_array(mfi);
            auto const &db_arr = Db.const_array(mfi);

            if (solve_dir < 0 || solve_dir > 2)
            {
                amrex::Abort("solve_dir must be 0, 1, or 2");
            }

            Box const b2d = amrex::makeSlab(bx, solve_dir, lo);
            Long const nlines = b2d.numPts();
            Gpu::DeviceVector<Real> line_work(nlines * nsolve * n_line_work);
            Gpu::DeviceVector<Real> line_coeff(nlines * nsolve * n_line_coeff);
            Real *work = line_work.data();
            Real *coeff = line_coeff.data();
            int const xlo = b2d.smallEnd(0);
            int const ylo = b2d.smallEnd(1);
            int const zlo = b2d.smallEnd(2);
            int const xlen = b2d.length(0);
            int const ylen = b2d.length(1);

            amrex::ParallelForOMP(b2d, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                  {
                if (skip_pec_lines &&
                    pecOnSolveSlab(pec_normal, pec_location, pec_lo, pec_hi,
                                   solve_dir, i, j, k))
                {
                    for (int s = lo; s <= hi; ++s)
                    {
                        setLineValue(field_arr, solve_dir, s, i, j, k, 0.0_rt);
                    }
                    return;
                }

                int const line_id =
                    (i - xlo) + (j - ylo) * xlen + (k - zlo) * xlen * ylen;
                Real *cprime = work + line_id * nsolve * n_line_work;
                Real *dprime = cprime + nsolve;
                Real *x = dprime + nsolve;
                Real *z = x + nsolve;
                Real *a = coeff + line_id * nsolve * n_line_coeff;
                Real *bb = a + nsolve;
                Real *c = bb + nsolve;

                Real db_seam = lineValue(db_arr, solve_dir, hi - 1, i, j, k);
                Real alpha_cyclic = -db_seam * inv_d2;
                Real beta_cyclic = -db_seam * inv_d2;

                for (int p = 0; p < nsolve; ++p)
                {
                    int const s = lo + p;

                    if (split_pec && p == iw)
                    {
                        a[p] = 0.0_rt;
                        bb[p] = 1.0_rt;
                        c[p] = 0.0_rt;
                        continue;
                    }

                    Real const db_lo = (p == 0)
                                           ? lineValue(db_arr, solve_dir, hi - 1, i, j, k)
                                           : lineValue(db_arr, solve_dir, s - 1, i, j, k);
                    Real const db_hi = lineValue(db_arr, solve_dir, s, i, j, k);

                    Real const al = db_lo * inv_d2;
                    Real const ga = db_hi * inv_d2;
                    bb[p] = 1.0_rt / lineValue(cb_arr, solve_dir, s, i, j, k) + al + ga;
                    a[p] = (p == 0) ? 0.0_rt : -al;
                    c[p] = (p == nsolve - 1) ? 0.0_rt : -ga;
                }

                Real gamma = -bb[0];
                bb[0] -= gamma;
                bb[nsolve - 1] -= alpha_cyclic * beta_cyclic / gamma;

                for (int p = 0; p < nsolve; ++p)
                {
                    int const s = lo + p;
                    x[p] = (split_pec && p == iw)
                               ? 0.0_rt
                               : lineValue(field_arr, solve_dir, s, i, j, k);
                }

                solveCyclicTridiagonal(a, bb, c, alpha_cyclic, beta_cyclic, gamma, x, x,
                                       cprime, dprime, z, nsolve);

                for (int p = 0; p < nsolve; ++p)
                {
                    setLineValue(field_arr, solve_dir, lo + p, i, j, k, x[p]);
                }
                if (split_pec)
                {
                    setLineValue(field_arr, solve_dir, interior_pec_iw, i, j, k, 0.0_rt);
                }
                setLineValue(field_arr, solve_dir, hi, i, j, k, x[0]); });
        }
    }

    void solvePeriodicCyclicLinesPml(MultiFab &field,
                                     MultiFab const &rhs,
                                     MultiFab const &Cb,
                                     MultiFab const &Db,
                                     MultiFab const &kappa,
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
        int const nsolve = hi - lo;
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

        constexpr int n_line_work = 4;  // cprime, dprime, x, z
        constexpr int n_line_coeff = 3; // a, bb, c per row

        for (MFIter mfi(field); mfi.isValid(); ++mfi)
        {
            Box const &bx = mfi.validbox();
            amrex::ignore_unused(solver_name);
            auto const &field_arr = field.array(mfi);
            auto const &cb_arr = Cb.const_array(mfi);
            auto const &db_arr = Db.const_array(mfi);
            auto const &k_arr = kappa.const_array(mfi);

            if (solve_dir < 0 || solve_dir > 2)
            {
                amrex::Abort("solve_dir must be 0, 1, or 2");
            }

            Box const b2d = amrex::makeSlab(bx, solve_dir, lo);
            Long const nlines = b2d.numPts();
            Gpu::DeviceVector<Real> line_work(nlines * nsolve * n_line_work);
            Gpu::DeviceVector<Real> line_coeff(nlines * nsolve * n_line_coeff);
            Real *work = line_work.data();
            Real *coeff = line_coeff.data();
            int const xlo = b2d.smallEnd(0);
            int const ylo = b2d.smallEnd(1);
            int const zlo = b2d.smallEnd(2);
            int const xlen = b2d.length(0);
            int const ylen = b2d.length(1);

            amrex::ParallelForOMP(b2d, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                  {
                if (skip_pec_lines &&
                    pecOnSolveSlab(pec_normal, pec_location, pec_lo, pec_hi,
                                   solve_dir, i, j, k))
                {
                    for (int s = lo; s <= hi; ++s)
                    {
                        setLineValue(field_arr, solve_dir, s, i, j, k, 0.0_rt);
                    }
                    return;
                }

                int const line_id =
                    (i - xlo) + (j - ylo) * xlen + (k - zlo) * xlen * ylen;
                Real *cprime = work + line_id * nsolve * n_line_work;
                Real *dprime = cprime + nsolve;
                Real *x = dprime + nsolve;
                Real *z = x + nsolve;
                Real *a = coeff + line_id * nsolve * n_line_coeff;
                Real *bb = a + nsolve;
                Real *c = bb + nsolve;

                Real const k0 = lineValue(k_arr, solve_dir, lo, i, j, k);
                Real const knm1 = lineValue(k_arr, solve_dir, hi - 1, i, j, k);
                Real const db_seam = lineValue(db_arr, solve_dir, hi - 1, i, j, k);
                Real const seam_scale = 1.0_rt / (k0 * knm1);
                Real alpha_cyclic = -db_seam * seam_scale * inv_d2;
                Real beta_cyclic = -db_seam * seam_scale * inv_d2;

                for (int p = 0; p < nsolve; ++p)
                {
                    int const s = lo + p;

                    if (split_pec && p == iw)
                    {
                        a[p] = 0.0_rt;
                        bb[p] = 1.0_rt;
                        c[p] = 0.0_rt;
                        continue;
                    }

                    Real const k_center = lineValue(k_arr, solve_dir, s, i, j, k);
                    Real const k_lo = (p == 0)
                                          ? lineValue(k_arr, solve_dir, hi - 1, i, j, k)
                                          : lineValue(k_arr, solve_dir, s - 1, i, j, k);
                    Real const k_hi = lineValue(k_arr, solve_dir, s, i, j, k);
                    Real const db_lo = (p == 0)
                                           ? lineValue(db_arr, solve_dir, hi - 1, i, j, k)
                                           : lineValue(db_arr, solve_dir, s - 1, i, j, k);
                    Real const db_hi = lineValue(db_arr, solve_dir, s, i, j, k);

                    Real const al = db_lo * inv_d2 / (k_center * k_lo);
                    Real const ga = db_hi * inv_d2 / (k_center * k_hi);
                    bb[p] = 1.0_rt / lineValue(cb_arr, solve_dir, s, i, j, k) + al + ga;
                    a[p] = (p == 0) ? 0.0_rt : -al;
                    c[p] = (p == nsolve - 1) ? 0.0_rt : -ga;
                }

                Real gamma = -bb[0];
                bb[0] -= gamma;
                bb[nsolve - 1] -= alpha_cyclic * beta_cyclic / gamma;

                for (int p = 0; p < nsolve; ++p)
                {
                    int const s = lo + p;
                    x[p] = (split_pec && p == iw)
                               ? 0.0_rt
                               : lineValue(field_arr, solve_dir, s, i, j, k);
                }

                solveCyclicTridiagonal(a, bb, c, alpha_cyclic, beta_cyclic, gamma, x, x,
                                       cprime, dprime, z, nsolve);

                for (int p = 0; p < nsolve; ++p)
                {
                    setLineValue(field_arr, solve_dir, lo + p, i, j, k, x[p]);
                }
                if (split_pec)
                {
                    setLineValue(field_arr, solve_dir, interior_pec_iw, i, j, k, 0.0_rt);
                }
                setLineValue(field_arr, solve_dir, hi, i, j, k, x[0]); });
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

        constexpr int n_line_work = 3;  // cprime, dprime, x
        constexpr int n_line_coeff = 3; // a, b, c per row

        for (MFIter mfi(field); mfi.isValid(); ++mfi)
        {
            Box const &bx = mfi.validbox();
            amrex::ignore_unused(solver_name);
            auto const &field_arr = field.array(mfi);
            auto const &cb_arr = Cb.const_array(mfi);
            auto const &db_arr = Db.const_array(mfi);

            if (solve_dir < 0 || solve_dir > 2)
            {
                amrex::Abort("solve_dir must be 0, 1, or 2");
            }

            Box const b2d = amrex::makeSlab(bx, solve_dir, lo + 1);
            Long const nlines = b2d.numPts();
            Gpu::DeviceVector<Real> line_work(nlines * nsolve * n_line_work);
            Gpu::DeviceVector<Real> line_coeff(nlines * nsolve * n_line_coeff);
            Real *work = line_work.data();
            Real *coeff = line_coeff.data();
            int const xlo = b2d.smallEnd(0);
            int const ylo = b2d.smallEnd(1);
            int const zlo = b2d.smallEnd(2);
            int const xlen = b2d.length(0);
            int const ylen = b2d.length(1);

            amrex::ParallelForOMP(b2d, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                  {
                int const line_id =
                    (i - xlo) + (j - ylo) * xlen + (k - zlo) * xlen * ylen;
                Real *cprime = work + line_id * nsolve * n_line_work;
                Real *dprime = cprime + nsolve;
                Real *x = dprime + nsolve;
                Real *a = coeff + line_id * nsolve * n_line_coeff;
                Real *b = a + nsolve;
                Real *c = b + nsolve;

                for (int p = 0; p < nsolve; ++p)
                {
                    int const s = lo + 1 + p;
                    Real const db_lo = lineValue(db_arr, solve_dir, s - 1, i, j, k);
                    Real const db_hi = lineValue(db_arr, solve_dir, s, i, j, k);
                    Real const al = db_lo * inv_d2;
                    Real const ga = db_hi * inv_d2;
                    b[p] = 1.0_rt / lineValue(cb_arr, solve_dir, s, i, j, k) + al + ga;
                    a[p] = (p == 0) ? 0.0_rt : -al;
                    c[p] = (p == nsolve - 1) ? 0.0_rt : -ga;
                    x[p] = lineValue(field_arr, solve_dir, s, i, j, k);
                }

                solveTridiagonal(a, b, c, x, x, cprime, dprime, nsolve);

                for (int p = 0; p < nsolve; ++p)
                {
                    setLineValue(field_arr, solve_dir, lo + 1 + p, i, j, k, x[p]);
                }
                setLineValue(field_arr, solve_dir, lo, i, j, k, 0.0_rt);
                setLineValue(field_arr, solve_dir, hi, i, j, k, 0.0_rt); });
        }
    }

    void solveDirichletNodalLinesPml(MultiFab &field,
                                     MultiFab const &rhs,
                                     MultiFab const &Cb,
                                     MultiFab const &Db,
                                     MultiFab const &kappa,
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

        constexpr int n_line_work = 3;  // cprime, dprime, x
        constexpr int n_line_coeff = 3; // a, b, c per row

        for (MFIter mfi(field); mfi.isValid(); ++mfi)
        {
            Box const &bx = mfi.validbox();
            amrex::ignore_unused(solver_name);
            auto const &field_arr = field.array(mfi);
            auto const &cb_arr = Cb.const_array(mfi);
            auto const &db_arr = Db.const_array(mfi);
            auto const &k_arr = kappa.const_array(mfi);

            if (solve_dir < 0 || solve_dir > 2)
            {
                amrex::Abort("solve_dir must be 0, 1, or 2");
            }

            Box const b2d = amrex::makeSlab(bx, solve_dir, lo + 1);
            Long const nlines = b2d.numPts();
            Gpu::DeviceVector<Real> line_work(nlines * nsolve * n_line_work);
            Gpu::DeviceVector<Real> line_coeff(nlines * nsolve * n_line_coeff);
            Real *work = line_work.data();
            Real *coeff = line_coeff.data();
            int const xlo = b2d.smallEnd(0);
            int const ylo = b2d.smallEnd(1);
            int const zlo = b2d.smallEnd(2);
            int const xlen = b2d.length(0);
            int const ylen = b2d.length(1);

            amrex::ParallelForOMP(b2d, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                  {
                int const line_id =
                    (i - xlo) + (j - ylo) * xlen + (k - zlo) * xlen * ylen;
                Real *cprime = work + line_id * nsolve * n_line_work;
                Real *dprime = cprime + nsolve;
                Real *x = dprime + nsolve;
                Real *a = coeff + line_id * nsolve * n_line_coeff;
                Real *b = a + nsolve;
                Real *c = b + nsolve;

                for (int p = 0; p < nsolve; ++p)
                {
                    int const s = lo + 1 + p;
                    Real const k_center = lineValue(k_arr, solve_dir, s, i, j, k);
                    Real const k_lo = lineValue(k_arr, solve_dir, s - 1, i, j, k);
                    Real const k_hi = lineValue(k_arr, solve_dir, s, i, j, k);
                    Real const db_lo = lineValue(db_arr, solve_dir, s - 1, i, j, k);
                    Real const db_hi = lineValue(db_arr, solve_dir, s, i, j, k);
                    Real const al = db_lo * inv_d2 / (k_center * k_lo);
                    Real const ga = db_hi * inv_d2 / (k_center * k_hi);
                    b[p] = 1.0_rt / lineValue(cb_arr, solve_dir, s, i, j, k) + al + ga;
                    a[p] = (p == 0) ? 0.0_rt : -al;
                    c[p] = (p == nsolve - 1) ? 0.0_rt : -ga;
                    x[p] = lineValue(field_arr, solve_dir, s, i, j, k);
                }

                solveTridiagonal(a, b, c, x, x, cprime, dprime, nsolve);

                for (int p = 0; p < nsolve; ++p)
                {
                    setLineValue(field_arr, solve_dir, lo + 1 + p, i, j, k, x[p]);
                }
                setLineValue(field_arr, solve_dir, lo, i, j, k, 0.0_rt);
                setLineValue(field_arr, solve_dir, hi, i, j, k, 0.0_rt); });
        }
    }

    void copyCoefToLayout(MultiFab &dst, MultiFab const &src, Periodicity const &period)
    {
        dst.ParallelCopy(src, 0, 0, 1, IntVect(0), dst.nGrowVect(), period);
    }

    void copyCoefToLayout(MultiFab &dst, MultiFab const &src,
                          Periodicity const &period, Real initial_value)
    {
        dst.setVal(initial_value);
        copyCoefToLayout(dst, src, period);
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

    m_pml_normal = -1;
    pp.query("pml_normal", m_pml_normal);
    if (m_pml_normal < -1 || m_pml_normal > 2)
    {
        amrex::Abort("adi.pml_normal must be -1 (off) or 0, 1, 2");
    }
    pp.query("pml_thickness", m_pml_thickness);
    pp.query("pml_sigma_max", m_pml_sigma_max);
    pp.query("pml_grade_m", m_pml_grade_m);
    pp.query("pml_alpha_max", m_pml_alpha_max);
    pp.query("pml_kappa_max", m_pml_kappa_max);
    m_pml_on = (m_pml_normal >= 0);
    if (m_pml_sigma_max < 0.0_rt)
    {
        amrex::Abort("adi.pml_sigma_max must be >= 0");
    }
    if (m_pml_alpha_max < 0.0_rt)
    {
        amrex::Abort("adi.pml_alpha_max must be >= 0");
    }
    if (m_pml_on)
    {
        if (m_pml_thickness <= 0)
        {
            amrex::Abort("adi.pml_thickness must be > 0 when adi.pml_normal >= 0");
        }
        if (m_pml_grade_m <= 0.0_rt)
        {
            amrex::Abort("adi.pml_grade_m must be > 0");
        }
        if (m_pml_kappa_max < 1.0_rt)
        {
            amrex::Abort("adi.pml_kappa_max must be >= 1");
        }
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
    if (m_pml_on)
    {
        if (2 * m_pml_thickness >= m_n_cells[m_pml_normal])
        {
            amrex::Abort("adi.pml_thickness is too large for adi.pml_normal extent");
        }
        if (m_pec_normal == m_pml_normal)
        {
            amrex::Abort("adi.pml_normal cannot match adi.pec_normal");
        }
    }

    RealBox real_box(prob_lo.begin(), prob_hi.begin());
    Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1, 1, 1)};
    if (m_pec_normal >= 0 && m_pec_location == 0)
    {
        is_periodic[m_pec_normal] = 0;
    }
    if (m_pml_on)
    {
        is_periodic[m_pml_normal] = 0;
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
        m_hfields[idim].define(amrex::convert(m_grids, btyp), m_dmap, 1, 1);
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
    initPmlProfiles();
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

void ADI::initData()
{
    InitSetupFields("adi", m_ic, m_ic_amplitude, m_ic_dir,
                    m_ic_pol, m_ic_wavelength, m_pulse_center, m_pulse_sigma,
                    m_eps_r, m_mu_r, m_geom, m_efields, m_hfields, true);
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
    updatePmlRecursionCoeffs(dt);

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
    FieldArray hfields_x, hfields_y, hfields_z;
    definePencilFields(efields_x, m_efields, bax, dmx);
    definePencilFields(efields_y, m_efields, bay, dmy);
    definePencilFields(efields_z, m_efields, baz, dmz);
    definePencilFields(hfields_x, m_hfields, bax, dmx);
    definePencilFields(hfields_y, m_hfields, bay, dmy);
    definePencilFields(hfields_z, m_hfields, baz, dmz);

    // Initialize ghost values once before stepping.
    {
        auto const period = m_geom.periodicity();
        Vector<MultiFab *> efield_ptrs{AMREX_D_DECL(&m_efields[0], &m_efields[1], &m_efields[2])};
        Vector<MultiFab *> hfield_ptrs{AMREX_D_DECL(&m_hfields[0], &m_hfields[1], &m_hfields[2])};
        amrex::FillBoundary(efield_ptrs, period);
        amrex::FillBoundary(hfield_ptrs, period);
        if (m_pml_on)
        {
            for (int idir = 0; idir < AMREX_SPACEDIM; ++idir)
            {
                m_efields[idir].setBndry(0.0_rt);
                m_hfields[idir].setBndry(0.0_rt);
            }
            PinPmlNodalBoundaryPlanes(m_pml_normal, m_efields);
            PinPmlNodalBoundaryPlanes(m_pml_normal, m_hfields);
        }
    }

    if (m_plot_int > 0)
    {
        UtilWritePlotOutput(m_plot_format, m_output_dir, 0, time,
                            m_conv_plt, m_ic_dir,
                            m_grids, m_dmap, m_geom, m_efields, m_hfields,
                            true);
    }

    for (int step = 0; step < m_max_step; ++step)
    {
        adiFirstHalfStep(m_efields, m_hfields, efields_x, efields_y,
                         efields_z, hfields_x, hfields_y, hfields_z, dt);
        adiSecondHalfStep(m_efields, m_hfields, efields_x, efields_y,
                          efields_z, hfields_x, hfields_y, hfields_z, dt);

        time += dt;

        if (m_plot_int > 0 && (step + 1) % m_plot_int == 0)
        {
            UtilWritePlotOutput(m_plot_format, m_output_dir, step + 1, time,
                                m_conv_plt, m_ic_dir,
                                m_grids, m_dmap, m_geom, m_efields, m_hfields,
                                true);
        }
    }
}

void ADI::adiFirstHalfStep(Array<MultiFab, AMREX_SPACEDIM> &efields,
                           Array<MultiFab, AMREX_SPACEDIM> &hfields,
                           FieldArray &efields_x, FieldArray &efields_y,
                           FieldArray &efields_z, FieldArray &hfields_x,
                           FieldArray &hfields_y, FieldArray &hfields_z,
                           Real dt)
{
    // eq:adi-first-half-amrex — implicit E along y,z,x; explicit H at n+1/2
    auto const period = m_geom.periodicity();
    Vector<MultiFab *> efield_ptrs{AMREX_D_DECL(&efields[0], &efields[1], &efields[2])};
    Vector<MultiFab *> hfield_ptrs{AMREX_D_DECL(&hfields[0], &hfields[1], &hfields[2])};

    Array<MultiFab, AMREX_SPACEDIM> eold = copyFieldsWithGhosts(efields);
    Array<MultiFab, AMREX_SPACEDIM> hold = copyFieldsWithGhosts(hfields);

    copyFields(efields_y, efields, period);
    copyFields(hfields_y, hfields, period);
    MultiFab rhs_ex = buildRhsEx1(efields_y, hfields_y, dt);

    copyFields(efields_z, efields, period);
    copyFields(hfields_z, hfields, period);
    MultiFab rhs_ey = buildRhsEy1(efields_z, hfields_z, dt);

    copyFields(efields_x, efields, period);
    copyFields(hfields_x, hfields, period);
    MultiFab rhs_ez = buildRhsEz1(efields_x, hfields_x, dt);

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
    if (m_pml_on)
    {
        for (int idir = 0; idir < AMREX_SPACEDIM; ++idir)
        {
            efields[idir].setBndry(0.0_rt);
        }
        PinPmlNodalBoundaryPlanes(m_pml_normal, efields);
    }

    PecPinTangentialE(m_pec_normal, m_pec_location, efields);

    if (m_pml_on)
    {
        stepHxPml1(hfields[0], efields[1], eold[2], dt);
        stepHyPml1(hfields[1], efields[2], eold[0], dt);
        stepHzPml1(hfields[2], efields[0], eold[1], dt);
    }
    else
    {
        stepHx(hfields[0], efields[1], eold[2], dt);
        stepHy(hfields[1], efields[2], eold[0], dt);
        stepHz(hfields[2], efields[0], eold[1], dt);
    }

    if (m_pml_on)
    {
        updatePmlAuxElectricFirstHalf(hold);
        updatePmlAuxMagneticFirstHalf(efields, eold);
    }

    amrex::FillBoundary(hfield_ptrs, period);
    if (m_pml_on)
    {
        for (int idir = 0; idir < AMREX_SPACEDIM; ++idir)
        {
            hfields[idir].setBndry(0.0_rt);
        }
        PinPmlNodalBoundaryPlanes(m_pml_normal, hfields);
    }
}

void ADI::adiSecondHalfStep(Array<MultiFab, AMREX_SPACEDIM> &efields,
                            Array<MultiFab, AMREX_SPACEDIM> &hfields,
                            FieldArray &efields_x, FieldArray &efields_y,
                            FieldArray &efields_z, FieldArray &hfields_x,
                            FieldArray &hfields_y, FieldArray &hfields_z,
                            Real dt)
{
    // eq:adi-second-half-amrex — implicit E along z,x,y; explicit H at n+1
    auto const period = m_geom.periodicity();
    Vector<MultiFab *> efield_ptrs{AMREX_D_DECL(&efields[0], &efields[1], &efields[2])};
    Vector<MultiFab *> hfield_ptrs{AMREX_D_DECL(&hfields[0], &hfields[1], &hfields[2])};

    Array<MultiFab, AMREX_SPACEDIM> eold = copyFieldsWithGhosts(efields);
    Array<MultiFab, AMREX_SPACEDIM> hold = copyFieldsWithGhosts(hfields);

    copyFields(efields_z, efields, period);
    copyFields(hfields_z, hfields, period);
    MultiFab rhs_ex = buildRhsEx2(efields_z, hfields_z, dt);

    copyFields(efields_x, efields, period);
    copyFields(hfields_x, hfields, period);
    MultiFab rhs_ey = buildRhsEy2(efields_x, hfields_x, dt);

    copyFields(efields_y, efields, period);
    copyFields(hfields_y, hfields, period);
    MultiFab rhs_ez = buildRhsEz2(efields_y, hfields_y, dt);

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
    if (m_pml_on)
    {
        for (int idir = 0; idir < AMREX_SPACEDIM; ++idir)
        {
            efields[idir].setBndry(0.0_rt);
        }
        PinPmlNodalBoundaryPlanes(m_pml_normal, efields);
    }

    PecPinTangentialE(m_pec_normal, m_pec_location, efields);

    if (m_pml_on)
    {
        stepHxPml2(hfields[0], eold[1], efields[2], dt);
        stepHyPml2(hfields[1], eold[2], efields[0], dt);
        stepHzPml2(hfields[2], eold[0], efields[1], dt);
    }
    else
    {
        stepHx(hfields[0], eold[1], efields[2], dt);
        stepHy(hfields[1], eold[2], efields[0], dt);
        stepHz(hfields[2], eold[0], efields[1], dt);
    }

    if (m_pml_on)
    {
        updatePmlAuxElectricSecondHalf(hold);
        updatePmlAuxMagneticSecondHalf(efields, eold);
    }

    amrex::FillBoundary(hfield_ptrs, period);
    if (m_pml_on)
    {
        for (int idir = 0; idir < AMREX_SPACEDIM; ++idir)
        {
            hfields[idir].setBndry(0.0_rt);
        }
        PinPmlNodalBoundaryPlanes(m_pml_normal, hfields);
    }
}

MultiFab ADI::buildRhsEx1(Array<MultiFab, AMREX_SPACEDIM> const &efields,
                          Array<MultiFab, AMREX_SPACEDIM> const &hfields,
                          Real dt) const
{
    amrex::ignore_unused(dt);
    // eq:first-e-ex-adi-tridiagonal — implicit along y.
    MultiFab rhs = makeRhsLike(efields[0]);
    MultiFab p_field = makeRhsLike(efields[0]);
    MultiFab db_field = makeCoeffLike(efields[0], m_Db[2]);
    auto const period = m_geom.periodicity();
    copyCoefToLayout(p_field, m_p[0], period);
    copyCoefToLayout(db_field, m_Db[2], period);

    auto const dx = m_geom.CellSizeArray();
    Real const dyinv = 1.0_rt / dx[1];
    Real const dzinv = 1.0_rt / dx[2];
    Real const dxinv = 1.0_rt / dx[0];

    if (!m_pml_on)
    {
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
            auto const &hz_arr = hfields[2].const_array(mfi);
            auto const &hy_arr = hfields[1].const_array(mfi);

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

    MultiFab ky_field = makeRhsLike(efields[0]);
    MultiFab kz_field = makeRhsLike(efields[0]);
    MultiFab kx_db = makeCoeffLike(efields[0], m_Db[2]);
    MultiFab psi_e0 = makeRhsLike(efields[0]);
    MultiFab psi_e1 = makeRhsLike(efields[0]);
    MultiFab psi_h0 = makeCoeffLike(efields[0], m_Db[2]);
    MultiFab psi_h1 = makeCoeffLike(efields[0], m_Db[2]);
    copyCoefToLayout(ky_field, m_kappa[1], period);
    copyCoefToLayout(kz_field, m_kappa[2], period);
    copyCoefToLayout(kx_db, m_kappa[0], period);
    copyCoefToLayout(psi_e0, m_psi_e[PSI_EXY], period);
    copyCoefToLayout(psi_e1, m_psi_e[PSI_EXZ], period);
    copyCoefToLayout(psi_h0, m_psi_h[PSI_HZY - PSI_HXY], period);
    copyCoefToLayout(psi_h1, m_psi_h[PSI_HZX - PSI_HXY], period);

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
        auto const &hz_arr = hfields[2].const_array(mfi);
        auto const &hy_arr = hfields[1].const_array(mfi);
        auto const &ky_arr = ky_field.const_array(mfi);
        auto const &kz_arr = kz_field.const_array(mfi);
        auto const &kx_arr = kx_db.const_array(mfi);
        auto const &pe0 = psi_e0.const_array(mfi);
        auto const &pe1 = psi_e1.const_array(mfi);
        auto const &ph0 = psi_h0.const_array(mfi);
        auto const &ph1 = psi_h1.const_array(mfi);

        ParallelFor(tilebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
            Real const q = db_arr(i, j - 1, k);
            Real const r = db_arr(i, j, k);
            Real const ky = ky_arr(i, j, k);
            Real const kz = kz_arr(i, j, k);
            Real const curl_h = (dyinv / ky) * (hz_arr(i, j, k) - hz_arr(i, j - 1, k)) -
                                (dzinv / kz) * (hy_arr(i, j, k) - hy_arr(i, j, k - 1));
            Real const ey_lo = ey_arr(i + 1, j - 1, k) - ey_arr(i, j - 1, k);
            Real const ey_hi = ey_arr(i + 1, j, k) - ey_arr(i, j, k);
            Real const kx_lo = kx_arr(i, j - 1, k);
            Real const kx_hi = kx_arr(i, j, k);
            Real const psi_h_hi = ph0(i, j, k) - ph1(i, j, k);
            Real const psi_h_lo = ph0(i, j - 1, k) - ph1(i, j - 1, k);
            Real const psi_h_term = -(r * dyinv / ky) * psi_h_hi +
                                    (q * dyinv / ky) * psi_h_lo;
            Real const psi_e_term = pe1(i, j, k) - pe0(i, j, k);
            rhs_arr(i, j, k) = p_arr(i, j, k) * ex_arr(i, j, k) + curl_h +
                               q * dxinv * dyinv * ey_lo / (ky * kx_lo) -
                               r * dxinv * dyinv * ey_hi / (ky * kx_hi) +
                               psi_h_term + psi_e_term; });
    }

    return rhs;
}

MultiFab ADI::buildRhsEy1(Array<MultiFab, AMREX_SPACEDIM> const &efields,
                          Array<MultiFab, AMREX_SPACEDIM> const &hfields,
                          Real dt) const
{
    amrex::ignore_unused(dt);
    // eq:first-e-ey-adi-tridiagonal — implicit along z.
    MultiFab rhs = makeRhsLike(efields[1]);
    MultiFab p_field = makeRhsLike(efields[1]);
    MultiFab db_field = makeCoeffLike(efields[1], m_Db[0]);
    auto const period = m_geom.periodicity();
    copyCoefToLayout(p_field, m_p[1], period);
    copyCoefToLayout(db_field, m_Db[0], period);

    auto const dx = m_geom.CellSizeArray();
    Real const dzinv = 1.0_rt / dx[2];
    Real const dxinv = 1.0_rt / dx[0];
    Real const dyinv = 1.0_rt / dx[1];

    if (!m_pml_on)
    {
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
            auto const &hx_arr = hfields[0].const_array(mfi);
            auto const &hz_arr = hfields[2].const_array(mfi);

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

    MultiFab kz_field = makeRhsLike(efields[1]);
    MultiFab kx_field = makeRhsLike(efields[1]);
    MultiFab ky_db = makeCoeffLike(efields[1], m_Db[0]);
    MultiFab psi_e0 = makeRhsLike(efields[1]);
    MultiFab psi_e1 = makeRhsLike(efields[1]);
    MultiFab psi_h0 = makeCoeffLike(efields[1], m_Db[0]);
    MultiFab psi_h1 = makeCoeffLike(efields[1], m_Db[0]);
    copyCoefToLayout(kz_field, m_kappa[2], period);
    copyCoefToLayout(kx_field, m_kappa[0], period);
    copyCoefToLayout(ky_db, m_kappa[1], period);
    copyCoefToLayout(psi_e0, m_psi_e[PSI_EYZ], period);
    copyCoefToLayout(psi_e1, m_psi_e[PSI_EYX], period);
    copyCoefToLayout(psi_h0, m_psi_h[PSI_HXZ - PSI_HXY], period);
    copyCoefToLayout(psi_h1, m_psi_h[PSI_HXY - PSI_HXY], period);

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
        auto const &hx_arr = hfields[0].const_array(mfi);
        auto const &hz_arr = hfields[2].const_array(mfi);
        auto const &kz_arr = kz_field.const_array(mfi);
        auto const &kx_arr = kx_field.const_array(mfi);
        auto const &ky_arr = ky_db.const_array(mfi);
        auto const &pe0 = psi_e0.const_array(mfi);
        auto const &pe1 = psi_e1.const_array(mfi);
        auto const &ph0 = psi_h0.const_array(mfi);
        auto const &ph1 = psi_h1.const_array(mfi);

        ParallelFor(tilebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
            Real const q = db_arr(i, j, k - 1);
            Real const r = db_arr(i, j, k);
            Real const kz = kz_arr(i, j, k);
            Real const kx = kx_arr(i, j, k);
            Real const curl_h = (dzinv / kz) * (hx_arr(i, j, k) - hx_arr(i, j, k - 1)) -
                                (dxinv / kx) * (hz_arr(i, j, k) - hz_arr(i - 1, j, k));
            Real const ez_lo = ez_arr(i, j + 1, k - 1) - ez_arr(i, j, k - 1);
            Real const ez_hi = ez_arr(i, j + 1, k) - ez_arr(i, j, k);
            Real const ky_lo = ky_arr(i, j, k - 1);
            Real const ky_hi = ky_arr(i, j, k);
            Real const psi_h_hi = ph0(i, j, k) - ph1(i, j, k);
            Real const psi_h_lo = ph0(i, j, k - 1) - ph1(i, j, k - 1);
            Real const psi_h_term = -(r * dzinv / kz) * psi_h_hi +
                                    (q * dzinv / kz) * psi_h_lo;
            Real const psi_e_term = pe1(i, j, k) - pe0(i, j, k);
            rhs_arr(i, j, k) = p_arr(i, j, k) * ey_arr(i, j, k) + curl_h +
                               q * dyinv * dzinv * ez_lo / (kz * ky_lo) -
                               r * dyinv * dzinv * ez_hi / (kz * ky_hi) +
                               psi_h_term + psi_e_term; });
    }

    return rhs;
}

MultiFab ADI::buildRhsEz1(Array<MultiFab, AMREX_SPACEDIM> const &efields,
                          Array<MultiFab, AMREX_SPACEDIM> const &hfields,
                          Real dt) const
{
    amrex::ignore_unused(dt);
    // eq:first-e-ez-adi-tridiagonal — implicit along x.
    MultiFab rhs = makeRhsLike(efields[2]);
    MultiFab p_field = makeRhsLike(efields[2]);
    MultiFab db_field = makeCoeffLike(efields[2], m_Db[1]);
    auto const period = m_geom.periodicity();
    copyCoefToLayout(p_field, m_p[2], period);
    copyCoefToLayout(db_field, m_Db[1], period);

    auto const dx = m_geom.CellSizeArray();
    Real const dxinv = 1.0_rt / dx[0];
    Real const dyinv = 1.0_rt / dx[1];
    Real const dzinv = 1.0_rt / dx[2];

    if (!m_pml_on)
    {
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
            auto const &hy_arr = hfields[1].const_array(mfi);
            auto const &hx_arr = hfields[0].const_array(mfi);

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

    MultiFab kx_field = makeRhsLike(efields[2]);
    MultiFab ky_field = makeRhsLike(efields[2]);
    MultiFab kz_db = makeCoeffLike(efields[2], m_Db[1]);
    MultiFab psi_e0 = makeRhsLike(efields[2]);
    MultiFab psi_e1 = makeRhsLike(efields[2]);
    MultiFab psi_h0 = makeCoeffLike(efields[2], m_Db[1]);
    MultiFab psi_h1 = makeCoeffLike(efields[2], m_Db[1]);
    copyCoefToLayout(kx_field, m_kappa[0], period);
    copyCoefToLayout(ky_field, m_kappa[1], period);
    copyCoefToLayout(kz_db, m_kappa[2], period);
    copyCoefToLayout(psi_e0, m_psi_e[PSI_EZX], period);
    copyCoefToLayout(psi_e1, m_psi_e[PSI_EZY], period);
    copyCoefToLayout(psi_h0, m_psi_h[PSI_HYX - PSI_HXY], period);
    copyCoefToLayout(psi_h1, m_psi_h[PSI_HYZ - PSI_HXY], period);

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
        auto const &hy_arr = hfields[1].const_array(mfi);
        auto const &hx_arr = hfields[0].const_array(mfi);
        auto const &kx_arr = kx_field.const_array(mfi);
        auto const &ky_arr = ky_field.const_array(mfi);
        auto const &kz_arr = kz_db.const_array(mfi);
        auto const &pe0 = psi_e0.const_array(mfi);
        auto const &pe1 = psi_e1.const_array(mfi);
        auto const &ph0 = psi_h0.const_array(mfi);
        auto const &ph1 = psi_h1.const_array(mfi);

        ParallelFor(tilebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
            Real const q = db_arr(i - 1, j, k);
            Real const r = db_arr(i, j, k);
            Real const kx = kx_arr(i, j, k);
            Real const ky = ky_arr(i, j, k);
            Real const curl_h = (dxinv / kx) * (hy_arr(i, j, k) - hy_arr(i - 1, j, k)) -
                                (dyinv / ky) * (hx_arr(i, j, k) - hx_arr(i, j - 1, k));
            Real const ex_lo = ex_arr(i - 1, j, k + 1) - ex_arr(i - 1, j, k);
            Real const ex_hi = ex_arr(i, j, k + 1) - ex_arr(i, j, k);
            Real const kz_lo = kz_arr(i - 1, j, k);
            Real const kz_hi = kz_arr(i, j, k);
            Real const psi_h_hi = ph0(i, j, k) - ph1(i, j, k);
            Real const psi_h_lo = ph0(i - 1, j, k) - ph1(i - 1, j, k);
            Real const psi_h_term = -(r * dxinv / kx) * psi_h_hi +
                                    (q * dxinv / kx) * psi_h_lo;
            Real const psi_e_term = pe1(i, j, k) - pe0(i, j, k);
            rhs_arr(i, j, k) = p_arr(i, j, k) * ez_arr(i, j, k) + curl_h +
                               q * dzinv * dxinv * ex_lo / (kx * kz_lo) -
                               r * dzinv * dxinv * ex_hi / (kx * kz_hi) +
                               psi_h_term + psi_e_term; });
    }

    return rhs;
}

void ADI::solveImplicitEx1(MultiFab &ex, MultiFab const &rhs, Real dt) const
{
    amrex::ignore_unused(dt);
    MultiFab Cb = makeRhsLike(ex);
    MultiFab Db = makeCoeffLike(ex, m_Db[2]);
    MultiFab Kappa = makeRhsLike(ex);
    auto const period = m_geom.periodicity();
    copyCoefToLayout(Cb, m_Cb[0], period);
    copyCoefToLayout(Db, m_Db[2], period);
    copyCoefToLayout(Kappa, m_kappa[1], period);
    Real const dy = m_geom.CellSizeArray()[1];
    Real const inv_d2 = 1.0_rt / (dy * dy);
    if (m_pml_on && m_pml_normal == 1)
    {
        solveDirichletNodalLinesPml(ex, rhs, Cb, Db, Kappa, 1, inv_d2, "solveImplicitEx1");
    }
    else if (m_pec_location == 0 && m_pec_normal == 1)
    {
        if (m_pml_on && m_pml_normal == 1)
        {
            solveDirichletNodalLinesPml(ex, rhs, Cb, Db, Kappa, 1, inv_d2, "solveImplicitEx1");
        }
        else
        {
            solveDirichletNodalLines(ex, rhs, Cb, Db, 1, inv_d2, "solveImplicitEx1");
        }
    }
    else if (m_pec_location != 0 && m_pec_normal == 1)
    {
        if (m_pml_on && m_pml_normal == 1)
        {
            solvePeriodicCyclicLinesPml(ex, rhs, Cb, Db, Kappa, 1, inv_d2, m_pec_location, -1, 0, -1,
                                        "solveImplicitEx1");
        }
        else
        {
            solvePeriodicCyclicLines(ex, rhs, Cb, Db, 1, inv_d2, m_pec_location, -1, 0, -1,
                                     "solveImplicitEx1");
        }
    }
    else
    {
        if (m_pml_on && m_pml_normal == 1)
        {
            solvePeriodicCyclicLinesPml(ex, rhs, Cb, Db, Kappa, 1, inv_d2, -1, m_pec_normal, m_pec_location, 0,
                                        "solveImplicitEx1");
        }
        else
        {
            solvePeriodicCyclicLines(ex, rhs, Cb, Db, 1, inv_d2, -1, m_pec_normal, m_pec_location, 0,
                                     "solveImplicitEx1");
        }
    }
}

void ADI::solveImplicitEy1(MultiFab &ey, MultiFab const &rhs, Real dt) const
{
    amrex::ignore_unused(dt);
    MultiFab Cb = makeRhsLike(ey);
    MultiFab Db = makeCoeffLike(ey, m_Db[0]);
    MultiFab Kappa = makeRhsLike(ey);
    auto const period = m_geom.periodicity();
    copyCoefToLayout(Cb, m_Cb[1], period);
    copyCoefToLayout(Db, m_Db[0], period);
    copyCoefToLayout(Kappa, m_kappa[2], period);
    Real const dz = m_geom.CellSizeArray()[2];
    Real const inv_d2 = 1.0_rt / (dz * dz);
    if (m_pml_on && m_pml_normal == 2)
    {
        solveDirichletNodalLinesPml(ey, rhs, Cb, Db, Kappa, 2, inv_d2, "solveImplicitEy1");
    }
    else if (m_pec_location == 0 && m_pec_normal == 2)
    {
        if (m_pml_on && m_pml_normal == 2)
        {
            solveDirichletNodalLinesPml(ey, rhs, Cb, Db, Kappa, 2, inv_d2, "solveImplicitEy1");
        }
        else
        {
            solveDirichletNodalLines(ey, rhs, Cb, Db, 2, inv_d2, "solveImplicitEy1");
        }
    }
    else if (m_pec_location != 0 && m_pec_normal == 2)
    {
        if (m_pml_on && m_pml_normal == 2)
        {
            solvePeriodicCyclicLinesPml(ey, rhs, Cb, Db, Kappa, 2, inv_d2, m_pec_location, -1, 0, -1,
                                        "solveImplicitEy1");
        }
        else
        {
            solvePeriodicCyclicLines(ey, rhs, Cb, Db, 2, inv_d2, m_pec_location, -1, 0, -1,
                                     "solveImplicitEy1");
        }
    }
    else
    {
        if (m_pml_on && m_pml_normal == 2)
        {
            solvePeriodicCyclicLinesPml(ey, rhs, Cb, Db, Kappa, 2, inv_d2, -1, m_pec_normal, m_pec_location, 1,
                                        "solveImplicitEy1");
        }
        else
        {
            solvePeriodicCyclicLines(ey, rhs, Cb, Db, 2, inv_d2, -1, m_pec_normal, m_pec_location, 1,
                                     "solveImplicitEy1");
        }
    }
}

void ADI::solveImplicitEz1(MultiFab &ez, MultiFab const &rhs, Real dt) const
{
    amrex::ignore_unused(dt);
    MultiFab Cb = makeRhsLike(ez);
    MultiFab Db = makeCoeffLike(ez, m_Db[1]);
    MultiFab Kappa = makeRhsLike(ez);
    auto const period = m_geom.periodicity();
    copyCoefToLayout(Cb, m_Cb[2], period);
    copyCoefToLayout(Db, m_Db[1], period);
    copyCoefToLayout(Kappa, m_kappa[0], period);
    Real const dx = m_geom.CellSizeArray()[0];
    Real const inv_d2 = 1.0_rt / (dx * dx);
    if (m_pml_on && m_pml_normal == 0)
    {
        solveDirichletNodalLinesPml(ez, rhs, Cb, Db, Kappa, 0, inv_d2, "solveImplicitEz1");
    }
    else if (m_pec_location == 0 && m_pec_normal == 0)
    {
        if (m_pml_on && m_pml_normal == 0)
        {
            solveDirichletNodalLinesPml(ez, rhs, Cb, Db, Kappa, 0, inv_d2, "solveImplicitEz1");
        }
        else
        {
            solveDirichletNodalLines(ez, rhs, Cb, Db, 0, inv_d2, "solveImplicitEz1");
        }
    }
    else if (m_pec_location != 0 && m_pec_normal == 0)
    {
        if (m_pml_on && m_pml_normal == 0)
        {
            solvePeriodicCyclicLinesPml(ez, rhs, Cb, Db, Kappa, 0, inv_d2, m_pec_location, -1, 0, -1,
                                        "solveImplicitEz1");
        }
        else
        {
            solvePeriodicCyclicLines(ez, rhs, Cb, Db, 0, inv_d2, m_pec_location, -1, 0, -1,
                                     "solveImplicitEz1");
        }
    }
    else
    {
        if (m_pml_on && m_pml_normal == 0)
        {
            solvePeriodicCyclicLinesPml(ez, rhs, Cb, Db, Kappa, 0, inv_d2, -1, m_pec_normal, m_pec_location, 2,
                                        "solveImplicitEz1");
        }
        else
        {
            solvePeriodicCyclicLines(ez, rhs, Cb, Db, 0, inv_d2, -1, m_pec_normal, m_pec_location, 2,
                                     "solveImplicitEz1");
        }
    }
}

MultiFab ADI::buildRhsEx2(Array<MultiFab, AMREX_SPACEDIM> const &efields,
                          Array<MultiFab, AMREX_SPACEDIM> const &hfields,
                          Real dt) const
{
    amrex::ignore_unused(dt);
    // eq:second-e-ex-adi-tridiagonal — implicit along z.
    MultiFab rhs = makeRhsLike(efields[0]);
    MultiFab p_field = makeRhsLike(efields[0]);
    MultiFab db_field = makeCoeffLike(efields[0], m_Db[1]);
    auto const period = m_geom.periodicity();
    copyCoefToLayout(p_field, m_p[0], period);
    copyCoefToLayout(db_field, m_Db[1], period);

    auto const dx = m_geom.CellSizeArray();
    Real const dyinv = 1.0_rt / dx[1];
    Real const dzinv = 1.0_rt / dx[2];
    Real const dxinv = 1.0_rt / dx[0];

    if (!m_pml_on)
    {
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
            auto const &hz_arr = hfields[2].const_array(mfi);
            auto const &hy_arr = hfields[1].const_array(mfi);

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

    MultiFab ky_field = makeRhsLike(efields[0]);
    MultiFab kz_field = makeRhsLike(efields[0]);
    MultiFab kx_db = makeCoeffLike(efields[0], m_Db[1]);
    MultiFab psi_e0 = makeRhsLike(efields[0]);
    MultiFab psi_e1 = makeRhsLike(efields[0]);
    MultiFab psi_h0 = makeCoeffLike(efields[0], m_Db[1]);
    MultiFab psi_h1 = makeCoeffLike(efields[0], m_Db[1]);
    copyCoefToLayout(ky_field, m_kappa[1], period);
    copyCoefToLayout(kz_field, m_kappa[2], period);
    copyCoefToLayout(kx_db, m_kappa[0], period);
    copyCoefToLayout(psi_e0, m_psi_e[PSI_EXY], period);
    copyCoefToLayout(psi_e1, m_psi_e[PSI_EXZ], period);
    copyCoefToLayout(psi_h0, m_psi_h[PSI_HYX - PSI_HXY], period);
    copyCoefToLayout(psi_h1, m_psi_h[PSI_HYZ - PSI_HXY], period);

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
        auto const &hz_arr = hfields[2].const_array(mfi);
        auto const &hy_arr = hfields[1].const_array(mfi);
        auto const &ky_arr = ky_field.const_array(mfi);
        auto const &kz_arr = kz_field.const_array(mfi);
        auto const &kx_arr = kx_db.const_array(mfi);
        auto const &pe0 = psi_e0.const_array(mfi);
        auto const &pe1 = psi_e1.const_array(mfi);
        auto const &ph0 = psi_h0.const_array(mfi);
        auto const &ph1 = psi_h1.const_array(mfi);

        ParallelFor(tilebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
            Real const q = db_arr(i, j, k - 1);
            Real const r = db_arr(i, j, k);
            Real const ky = ky_arr(i, j, k);
            Real const kz = kz_arr(i, j, k);
            Real const curl_h = (dyinv / ky) * (hz_arr(i, j, k) - hz_arr(i, j - 1, k)) -
                                (dzinv / kz) * (hy_arr(i, j, k) - hy_arr(i, j, k - 1));
            Real const ez_lo = ez_arr(i + 1, j, k - 1) - ez_arr(i, j, k - 1);
            Real const ez_hi = ez_arr(i + 1, j, k) - ez_arr(i, j, k);
            Real const kx_lo = kx_arr(i, j, k - 1);
            Real const kx_hi = kx_arr(i, j, k);
            Real const psi_h_hi = ph0(i, j, k) - ph1(i, j, k);
            Real const psi_h_lo = ph0(i, j, k - 1) - ph1(i, j, k - 1);
            Real const psi_h_term = -(r * dzinv / kz) * psi_h_hi +
                                    (q * dzinv / kz) * psi_h_lo;
            Real const psi_e_term = pe1(i, j, k) - pe0(i, j, k);
            rhs_arr(i, j, k) = p_arr(i, j, k) * ex_arr(i, j, k) + curl_h +
                               q * dxinv * dzinv * ez_lo / (kz * kx_lo) -
                               r * dxinv * dzinv * ez_hi / (kz * kx_hi) +
                               psi_h_term + psi_e_term; });
    }

    return rhs;
}

MultiFab ADI::buildRhsEy2(Array<MultiFab, AMREX_SPACEDIM> const &efields,
                          Array<MultiFab, AMREX_SPACEDIM> const &hfields,
                          Real dt) const
{
    amrex::ignore_unused(dt);
    // eq:second-e-ey-adi-tridiagonal — implicit along x.
    MultiFab rhs = makeRhsLike(efields[1]);
    MultiFab p_field = makeRhsLike(efields[1]);
    MultiFab db_field = makeCoeffLike(efields[1], m_Db[2]);
    auto const period = m_geom.periodicity();
    copyCoefToLayout(p_field, m_p[1], period);
    copyCoefToLayout(db_field, m_Db[2], period);

    auto const dx = m_geom.CellSizeArray();
    Real const dzinv = 1.0_rt / dx[2];
    Real const dxinv = 1.0_rt / dx[0];
    Real const dyinv = 1.0_rt / dx[1];

    if (!m_pml_on)
    {
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
            auto const &hx_arr = hfields[0].const_array(mfi);
            auto const &hz_arr = hfields[2].const_array(mfi);

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

    MultiFab kz_field = makeRhsLike(efields[1]);
    MultiFab kx_field = makeRhsLike(efields[1]);
    MultiFab ky_db = makeCoeffLike(efields[1], m_Db[2]);
    MultiFab psi_e0 = makeRhsLike(efields[1]);
    MultiFab psi_e1 = makeRhsLike(efields[1]);
    MultiFab psi_h0 = makeCoeffLike(efields[1], m_Db[2]);
    MultiFab psi_h1 = makeCoeffLike(efields[1], m_Db[2]);
    copyCoefToLayout(kz_field, m_kappa[2], period);
    copyCoefToLayout(kx_field, m_kappa[0], period);
    copyCoefToLayout(ky_db, m_kappa[1], period);
    copyCoefToLayout(psi_e0, m_psi_e[PSI_EYZ], period);
    copyCoefToLayout(psi_e1, m_psi_e[PSI_EYX], period);
    copyCoefToLayout(psi_h0, m_psi_h[PSI_HZY - PSI_HXY], period);
    copyCoefToLayout(psi_h1, m_psi_h[PSI_HZX - PSI_HXY], period);

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
        auto const &hx_arr = hfields[0].const_array(mfi);
        auto const &hz_arr = hfields[2].const_array(mfi);
        auto const &kz_arr = kz_field.const_array(mfi);
        auto const &kx_arr = kx_field.const_array(mfi);
        auto const &ky_arr = ky_db.const_array(mfi);
        auto const &pe0 = psi_e0.const_array(mfi);
        auto const &pe1 = psi_e1.const_array(mfi);
        auto const &ph0 = psi_h0.const_array(mfi);
        auto const &ph1 = psi_h1.const_array(mfi);

        ParallelFor(tilebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
            Real const q = db_arr(i - 1, j, k);
            Real const r = db_arr(i, j, k);
            Real const kz = kz_arr(i, j, k);
            Real const kx = kx_arr(i, j, k);
            Real const curl_h = (dzinv / kz) * (hx_arr(i, j, k) - hx_arr(i, j, k - 1)) -
                                (dxinv / kx) * (hz_arr(i, j, k) - hz_arr(i - 1, j, k));
            Real const ex_lo = ex_arr(i - 1, j + 1, k) - ex_arr(i - 1, j, k);
            Real const ex_hi = ex_arr(i, j + 1, k) - ex_arr(i, j, k);
            Real const ky_lo = ky_arr(i - 1, j, k);
            Real const ky_hi = ky_arr(i, j, k);
            Real const psi_h_hi = ph0(i, j, k) - ph1(i, j, k);
            Real const psi_h_lo = ph0(i - 1, j, k) - ph1(i - 1, j, k);
            Real const psi_h_term = -(r * dxinv / kx) * psi_h_hi +
                                    (q * dxinv / kx) * psi_h_lo;
            Real const psi_e_term = pe1(i, j, k) - pe0(i, j, k);
            rhs_arr(i, j, k) = p_arr(i, j, k) * ey_arr(i, j, k) + curl_h +
                               q * dyinv * dxinv * ex_lo / (kx * ky_lo) -
                               r * dyinv * dxinv * ex_hi / (kx * ky_hi) +
                               psi_h_term + psi_e_term; });
    }

    return rhs;
}

MultiFab ADI::buildRhsEz2(Array<MultiFab, AMREX_SPACEDIM> const &efields,
                          Array<MultiFab, AMREX_SPACEDIM> const &hfields,
                          Real dt) const
{
    amrex::ignore_unused(dt);
    // eq:second-e-ez-adi-tridiagonal — implicit along y.
    MultiFab rhs = makeRhsLike(efields[2]);
    MultiFab p_field = makeRhsLike(efields[2]);
    MultiFab db_field = makeCoeffLike(efields[2], m_Db[0]);
    auto const period = m_geom.periodicity();
    copyCoefToLayout(p_field, m_p[2], period);
    copyCoefToLayout(db_field, m_Db[0], period);

    auto const dx = m_geom.CellSizeArray();
    Real const dxinv = 1.0_rt / dx[0];
    Real const dyinv = 1.0_rt / dx[1];
    Real const dzinv = 1.0_rt / dx[2];

    if (!m_pml_on)
    {
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
            auto const &hy_arr = hfields[1].const_array(mfi);
            auto const &hx_arr = hfields[0].const_array(mfi);

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

    MultiFab kx_field = makeRhsLike(efields[2]);
    MultiFab ky_field = makeRhsLike(efields[2]);
    MultiFab kz_db = makeCoeffLike(efields[2], m_Db[0]);
    MultiFab psi_e0 = makeRhsLike(efields[2]);
    MultiFab psi_e1 = makeRhsLike(efields[2]);
    MultiFab psi_h0 = makeCoeffLike(efields[2], m_Db[0]);
    MultiFab psi_h1 = makeCoeffLike(efields[2], m_Db[0]);
    copyCoefToLayout(kx_field, m_kappa[0], period);
    copyCoefToLayout(ky_field, m_kappa[1], period);
    copyCoefToLayout(kz_db, m_kappa[2], period);
    copyCoefToLayout(psi_e0, m_psi_e[PSI_EZX], period);
    copyCoefToLayout(psi_e1, m_psi_e[PSI_EZY], period);
    copyCoefToLayout(psi_h0, m_psi_h[PSI_HXZ - PSI_HXY], period);
    copyCoefToLayout(psi_h1, m_psi_h[PSI_HXY - PSI_HXY], period);

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
        auto const &hy_arr = hfields[1].const_array(mfi);
        auto const &hx_arr = hfields[0].const_array(mfi);
        auto const &kx_arr = kx_field.const_array(mfi);
        auto const &ky_arr = ky_field.const_array(mfi);
        auto const &kz_arr = kz_db.const_array(mfi);
        auto const &pe0 = psi_e0.const_array(mfi);
        auto const &pe1 = psi_e1.const_array(mfi);
        auto const &ph0 = psi_h0.const_array(mfi);
        auto const &ph1 = psi_h1.const_array(mfi);

        ParallelFor(tilebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
            Real const q = db_arr(i, j - 1, k);
            Real const r = db_arr(i, j, k);
            Real const kx = kx_arr(i, j, k);
            Real const ky = ky_arr(i, j, k);
            Real const curl_h = (dxinv / kx) * (hy_arr(i, j, k) - hy_arr(i - 1, j, k)) -
                                (dyinv / ky) * (hx_arr(i, j, k) - hx_arr(i, j - 1, k));
            Real const ey_lo = ey_arr(i, j - 1, k + 1) - ey_arr(i, j - 1, k);
            Real const ey_hi = ey_arr(i, j, k + 1) - ey_arr(i, j, k);
            Real const kz_lo = kz_arr(i, j - 1, k);
            Real const kz_hi = kz_arr(i, j, k);
            Real const psi_h_hi = ph0(i, j, k) - ph1(i, j, k);
            Real const psi_h_lo = ph0(i, j - 1, k) - ph1(i, j - 1, k);
            Real const psi_h_term = -(r * dyinv / ky) * psi_h_hi +
                                    (q * dyinv / ky) * psi_h_lo;
            Real const psi_e_term = pe1(i, j, k) - pe0(i, j, k);
            rhs_arr(i, j, k) = p_arr(i, j, k) * ez_arr(i, j, k) + curl_h +
                               q * dzinv * dyinv * ey_lo / (ky * kz_lo) -
                               r * dzinv * dyinv * ey_hi / (ky * kz_hi) +
                               psi_h_term + psi_e_term; });
    }

    return rhs;
}

void ADI::solveImplicitEx2(MultiFab &ex, MultiFab const &rhs, Real dt) const
{
    amrex::ignore_unused(dt);
    MultiFab Cb = makeRhsLike(ex);
    MultiFab Db = makeCoeffLike(ex, m_Db[1]);
    MultiFab Kappa = makeRhsLike(ex);
    auto const period = m_geom.periodicity();
    copyCoefToLayout(Cb, m_Cb[0], period);
    copyCoefToLayout(Db, m_Db[1], period);
    copyCoefToLayout(Kappa, m_kappa[2], period);
    Real const dz = m_geom.CellSizeArray()[2];
    Real const inv_d2 = 1.0_rt / (dz * dz);
    if (m_pml_on && m_pml_normal == 2)
    {
        solveDirichletNodalLinesPml(ex, rhs, Cb, Db, Kappa, 2, inv_d2, "solveImplicitEx2");
    }
    else if (m_pec_location == 0 && m_pec_normal == 2)
    {
        if (m_pml_on && m_pml_normal == 2)
        {
            solveDirichletNodalLinesPml(ex, rhs, Cb, Db, Kappa, 2, inv_d2, "solveImplicitEx2");
        }
        else
        {
            solveDirichletNodalLines(ex, rhs, Cb, Db, 2, inv_d2, "solveImplicitEx2");
        }
    }
    else if (m_pec_location != 0 && m_pec_normal == 2)
    {
        if (m_pml_on && m_pml_normal == 2)
        {
            solvePeriodicCyclicLinesPml(ex, rhs, Cb, Db, Kappa, 2, inv_d2, m_pec_location, -1, 0, -1,
                                        "solveImplicitEx2");
        }
        else
        {
            solvePeriodicCyclicLines(ex, rhs, Cb, Db, 2, inv_d2, m_pec_location, -1, 0, -1,
                                     "solveImplicitEx2");
        }
    }
    else
    {
        if (m_pml_on && m_pml_normal == 2)
        {
            solvePeriodicCyclicLinesPml(ex, rhs, Cb, Db, Kappa, 2, inv_d2, -1, m_pec_normal, m_pec_location, 0,
                                        "solveImplicitEx2");
        }
        else
        {
            solvePeriodicCyclicLines(ex, rhs, Cb, Db, 2, inv_d2, -1, m_pec_normal, m_pec_location, 0,
                                     "solveImplicitEx2");
        }
    }
}

void ADI::solveImplicitEy2(MultiFab &ey, MultiFab const &rhs, Real dt) const
{
    amrex::ignore_unused(dt);
    MultiFab Cb = makeRhsLike(ey);
    MultiFab Db = makeCoeffLike(ey, m_Db[2]);
    MultiFab Kappa = makeRhsLike(ey);
    auto const period = m_geom.periodicity();
    copyCoefToLayout(Cb, m_Cb[1], period);
    copyCoefToLayout(Db, m_Db[2], period);
    copyCoefToLayout(Kappa, m_kappa[0], period);
    Real const dx = m_geom.CellSizeArray()[0];
    Real const inv_d2 = 1.0_rt / (dx * dx);
    if (m_pml_on && m_pml_normal == 0)
    {
        solveDirichletNodalLinesPml(ey, rhs, Cb, Db, Kappa, 0, inv_d2, "solveImplicitEy2");
    }
    else if (m_pec_location == 0 && m_pec_normal == 0)
    {
        if (m_pml_on && m_pml_normal == 0)
        {
            solveDirichletNodalLinesPml(ey, rhs, Cb, Db, Kappa, 0, inv_d2, "solveImplicitEy2");
        }
        else
        {
            solveDirichletNodalLines(ey, rhs, Cb, Db, 0, inv_d2, "solveImplicitEy2");
        }
    }
    else if (m_pec_location != 0 && m_pec_normal == 0)
    {
        if (m_pml_on && m_pml_normal == 0)
        {
            solvePeriodicCyclicLinesPml(ey, rhs, Cb, Db, Kappa, 0, inv_d2, m_pec_location, -1, 0, -1,
                                        "solveImplicitEy2");
        }
        else
        {
            solvePeriodicCyclicLines(ey, rhs, Cb, Db, 0, inv_d2, m_pec_location, -1, 0, -1,
                                     "solveImplicitEy2");
        }
    }
    else
    {
        if (m_pml_on && m_pml_normal == 0)
        {
            solvePeriodicCyclicLinesPml(ey, rhs, Cb, Db, Kappa, 0, inv_d2, -1, m_pec_normal, m_pec_location, 1,
                                        "solveImplicitEy2");
        }
        else
        {
            solvePeriodicCyclicLines(ey, rhs, Cb, Db, 0, inv_d2, -1, m_pec_normal, m_pec_location, 1,
                                     "solveImplicitEy2");
        }
    }
}

void ADI::solveImplicitEz2(MultiFab &ez, MultiFab const &rhs, Real dt) const
{
    amrex::ignore_unused(dt);
    MultiFab Cb = makeRhsLike(ez);
    MultiFab Db = makeCoeffLike(ez, m_Db[0]);
    MultiFab Kappa = makeRhsLike(ez);
    auto const period = m_geom.periodicity();
    copyCoefToLayout(Cb, m_Cb[2], period);
    copyCoefToLayout(Db, m_Db[0], period);
    copyCoefToLayout(Kappa, m_kappa[1], period);
    Real const dy = m_geom.CellSizeArray()[1];
    Real const inv_d2 = 1.0_rt / (dy * dy);
    if (m_pml_on && m_pml_normal == 1)
    {
        solveDirichletNodalLinesPml(ez, rhs, Cb, Db, Kappa, 1, inv_d2, "solveImplicitEz2");
    }
    else if (m_pec_location == 0 && m_pec_normal == 1)
    {
        if (m_pml_on && m_pml_normal == 1)
        {
            solveDirichletNodalLinesPml(ez, rhs, Cb, Db, Kappa, 1, inv_d2, "solveImplicitEz2");
        }
        else
        {
            solveDirichletNodalLines(ez, rhs, Cb, Db, 1, inv_d2, "solveImplicitEz2");
        }
    }
    else if (m_pec_location != 0 && m_pec_normal == 1)
    {
        if (m_pml_on && m_pml_normal == 1)
        {
            solvePeriodicCyclicLinesPml(ez, rhs, Cb, Db, Kappa, 1, inv_d2, m_pec_location, -1, 0, -1,
                                        "solveImplicitEz2");
        }
        else
        {
            solvePeriodicCyclicLines(ez, rhs, Cb, Db, 1, inv_d2, m_pec_location, -1, 0, -1,
                                     "solveImplicitEz2");
        }
    }
    else
    {
        if (m_pml_on && m_pml_normal == 1)
        {
            solvePeriodicCyclicLinesPml(ez, rhs, Cb, Db, Kappa, 1, inv_d2, -1, m_pec_normal, m_pec_location, 2,
                                        "solveImplicitEz2");
        }
        else
        {
            solvePeriodicCyclicLines(ez, rhs, Cb, Db, 1, inv_d2, -1, m_pec_normal, m_pec_location, 2,
                                     "solveImplicitEz2");
        }
    }
}

void ADI::stepHx(MultiFab &hx_dst, MultiFab const &ey_src,
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

void ADI::stepHy(MultiFab &hy_dst, MultiFab const &ez_src,
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

void ADI::stepHz(MultiFab &hz_dst, MultiFab const &ex_src,
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
