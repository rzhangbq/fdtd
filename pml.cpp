#include "adi.H"
#include "pml.H"

#include <AMReX_Gpu.H>

#include <cmath>

using namespace amrex;
namespace
{
    constexpr Real eps0 = 8.854187817e-12;
} // namespace

void ADI::initPmlProfiles()
{
    auto const period = m_geom.periodicity();
    auto const problo = m_geom.ProbLoArray();
    auto const probhi = m_geom.ProbHiArray();
    auto const dx = m_geom.CellSizeArray();
    int const pml_dir = m_pml_normal;
    Real const sigma_max = m_pml_sigma_max;
    Real const grade_m = m_pml_grade_m;
    Real const alpha_max = m_pml_alpha_max;
    Real const kappa_max = m_pml_kappa_max;
    int const pml_thickness = m_pml_thickness;
    bool const pml_on = m_pml_on;
    Real const zero = 0.0_rt;
    Real const one = 1.0_rt;

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        m_kappa[idim].define(m_grids, m_dmap, 1, 1);
        m_a_pml[idim].define(m_grids, m_dmap, 1, 1);
        m_b_pml[idim].define(m_grids, m_dmap, 1, 1);
        m_kappa[idim].setVal(1.0_rt);
        m_a_pml[idim].setVal(0.0_rt);
        m_b_pml[idim].setVal(1.0_rt);

        auto const &kappa_arr = m_kappa[idim].arrays();
        auto const &a_arr = m_a_pml[idim].arrays();
        auto const &b_arr = m_b_pml[idim].arrays();
        int const idir = idim;

        ParallelFor(m_kappa[idim], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                    {
            kappa_arr[b](i, j, k) = one;
            a_arr[b](i, j, k) = zero;
            b_arr[b](i, j, k) = one;

            if (!pml_on || idir != pml_dir)
            {
                return;
            }

            int idx = (idir == 0) ? i : ((idir == 1) ? j : k);
            Real const coord = problo[idir] + (static_cast<Real>(idx) + 0.5_rt) * dx[idir];
            Real const pml_len = static_cast<Real>(pml_thickness) * dx[idir];
            Real const inner_lo = problo[idir] + pml_len;
            Real const inner_hi = probhi[idir] - pml_len;

            Real rho = 0.0_rt;
            if (coord < inner_lo)
            {
                rho = inner_lo - coord;
            }
            else if (coord > inner_hi)
            {
                rho = coord - inner_hi;
            }

            if (rho <= 0.0_rt || pml_len <= 0.0_rt)
            {
                return;
            }

            Real frac = rho / pml_len;
            frac = amrex::max(0.0_rt, amrex::min(1.0_rt, frac));
            Real const poly = std::pow(frac, grade_m);
            kappa_arr[b](i, j, k) = 1.0_rt + (kappa_max - 1.0_rt) * poly;
            amrex::ignore_unused(alpha_max, sigma_max); });
    }

    Vector<MultiFab *> kappa_ptrs{AMREX_D_DECL(&m_kappa[0], &m_kappa[1], &m_kappa[2])};
    Vector<MultiFab *> a_ptrs{AMREX_D_DECL(&m_a_pml[0], &m_a_pml[1], &m_a_pml[2])};
    Vector<MultiFab *> b_ptrs{AMREX_D_DECL(&m_b_pml[0], &m_b_pml[1], &m_b_pml[2])};
    amrex::FillBoundary(kappa_ptrs, period);
    amrex::FillBoundary(a_ptrs, period);
    amrex::FillBoundary(b_ptrs, period);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        m_kappa[idim].setBndry(1.0_rt);
        m_a_pml[idim].setBndry(0.0_rt);
        m_b_pml[idim].setBndry(1.0_rt);
    }

    m_psi_e[PSI_EXY].define(m_efields[0].boxArray(), m_dmap, 1, m_efields[0].nGrowVect());
    m_psi_e[PSI_EXZ].define(m_efields[0].boxArray(), m_dmap, 1, m_efields[0].nGrowVect());
    m_psi_e[PSI_EYX].define(m_efields[1].boxArray(), m_dmap, 1, m_efields[1].nGrowVect());
    m_psi_e[PSI_EYZ].define(m_efields[1].boxArray(), m_dmap, 1, m_efields[1].nGrowVect());
    m_psi_e[PSI_EZX].define(m_efields[2].boxArray(), m_dmap, 1, m_efields[2].nGrowVect());
    m_psi_e[PSI_EZY].define(m_efields[2].boxArray(), m_dmap, 1, m_efields[2].nGrowVect());

    m_psi_h[PSI_HXY - PSI_HXY].define(m_hfields[0].boxArray(), m_dmap, 1, m_hfields[0].nGrowVect());
    m_psi_h[PSI_HXZ - PSI_HXY].define(m_hfields[0].boxArray(), m_dmap, 1, m_hfields[0].nGrowVect());
    m_psi_h[PSI_HYX - PSI_HXY].define(m_hfields[1].boxArray(), m_dmap, 1, m_hfields[1].nGrowVect());
    m_psi_h[PSI_HYZ - PSI_HXY].define(m_hfields[1].boxArray(), m_dmap, 1, m_hfields[1].nGrowVect());
    m_psi_h[PSI_HZX - PSI_HXY].define(m_hfields[2].boxArray(), m_dmap, 1, m_hfields[2].nGrowVect());
    m_psi_h[PSI_HZY - PSI_HXY].define(m_hfields[2].boxArray(), m_dmap, 1, m_hfields[2].nGrowVect());

    for (int ip = 0; ip < 6; ++ip)
    {
        m_psi_e[ip].setVal(0.0_rt);
        m_psi_h[ip].setVal(0.0_rt);
    }
}

void ADI::updatePmlRecursionCoeffs(Real dt)
{
    Real const dt_half = 0.5_rt * dt;
    auto const &kappa = m_kappa;
    auto &a = m_a_pml;
    auto &b = m_b_pml;
    auto const problo = m_geom.ProbLoArray();
    auto const probhi = m_geom.ProbHiArray();
    auto const dx = m_geom.CellSizeArray();
    int const pml_dir = m_pml_normal;
    bool const pml_on = m_pml_on;
    Real const sigma_max = m_pml_sigma_max;
    Real const alpha_max = m_pml_alpha_max;
    Real const grade_m = m_pml_grade_m;
    int const pml_thickness = m_pml_thickness;

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        auto const &kappa_arr = kappa[idim].const_arrays();
        auto const &a_arr = a[idim].arrays();
        auto const &b_arr = b[idim].arrays();
        int const idir = idim;

        ParallelFor(a[idim], [=] AMREX_GPU_DEVICE(int bidx, int i, int j, int k)
                    {
            if (!pml_on || idir != pml_dir)
            {
                a_arr[bidx](i, j, k) = 0.0_rt;
                b_arr[bidx](i, j, k) = 1.0_rt;
                return;
            }

            int idx = (idir == 0) ? i : ((idir == 1) ? j : k);
            Real const coord = problo[idir] + (static_cast<Real>(idx) + 0.5_rt) * dx[idir];
            Real const pml_len = static_cast<Real>(pml_thickness) * dx[idir];
            Real const inner_lo = problo[idir] + pml_len;
            Real const inner_hi = probhi[idir] - pml_len;

            Real rho = 0.0_rt;
            if (coord < inner_lo)
            {
                rho = inner_lo - coord;
            }
            else if (coord > inner_hi)
            {
                rho = coord - inner_hi;
            }

            if (rho <= 0.0_rt || pml_len <= 0.0_rt)
            {
                a_arr[bidx](i, j, k) = 0.0_rt;
                b_arr[bidx](i, j, k) = 1.0_rt;
                return;
            }

            Real frac = rho / pml_len;
            frac = amrex::max(0.0_rt, amrex::min(1.0_rt, frac));
            Real const poly = std::pow(frac, grade_m);
            Real const sig = sigma_max * poly;
            Real const alp = alpha_max * (1.0_rt - frac);
            Real const kapp = kappa_arr[bidx](i, j, k);

            Real const expo = -((sig / kapp) + alp) * dt_half / eps0;
            Real const bval = std::exp(expo);
            b_arr[bidx](i, j, k) = bval;

            Real const denom = kapp * (sig + kapp * alp);
            if (std::abs(denom) > 0.0_rt)
            {
                a_arr[bidx](i, j, k) = (sig / denom) * (bval - 1.0_rt);
            }
            else
            {
                a_arr[bidx](i, j, k) = 0.0_rt;
            } });
    }

    auto const period = m_geom.periodicity();
    Vector<MultiFab *> a_ptrs{AMREX_D_DECL(&m_a_pml[0], &m_a_pml[1], &m_a_pml[2])};
    Vector<MultiFab *> b_ptrs{AMREX_D_DECL(&m_b_pml[0], &m_b_pml[1], &m_b_pml[2])};
    amrex::FillBoundary(a_ptrs, period);
    amrex::FillBoundary(b_ptrs, period);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        m_a_pml[idim].setBndry(0.0_rt);
        m_b_pml[idim].setBndry(1.0_rt);
    }
}

void ADI::updatePmlAuxElectricFirstHalf(FieldArray const &hfields_n)
{
    auto const period = m_geom.periodicity();
    auto const dxinv = m_geom.InvCellSizeArray();
    auto update = [&](MultiFab &psi_dst, int deriv_dir, MultiFab const &num_hi,
                      MultiFab const &num_lo, IntVect const &off,
                      bool is_electric) {
        amrex::ignore_unused(is_electric);
        MultiFab acoef = PmlMakeCoeffLike(psi_dst, 0.0_rt);
        MultiFab bcoef = PmlMakeCoeffLike(psi_dst, 1.0_rt);
        PmlCopyCoefToLayout(acoef, m_a_pml[deriv_dir], period);
        PmlCopyCoefToLayout(bcoef, m_b_pml[deriv_dir], period);
        auto const &psi = psi_dst.arrays();
        auto const &aa = acoef.const_arrays();
        auto const &bb = bcoef.const_arrays();
        auto const &hi = num_hi.const_arrays();
        auto const &lo = num_lo.const_arrays();
        Real const dinv = dxinv[deriv_dir];
        ParallelFor(psi_dst, [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                    {
            int const ii = i + off[0];
            int const jj = j + off[1];
            int const kk = k + off[2];
            // Store CPML corrections with the sign used by the ADI RHS terms.
            Real const dnum = lo[b](ii, jj, kk) - hi[b](i, j, k);
            psi[b](i, j, k) = bb[b](i, j, k) * psi[b](i, j, k) +
                              aa[b](i, j, k) * dinv * dnum; });
    };

    update(m_psi_e[PSI_EXY], 1, hfields_n[2], hfields_n[2], IntVect(0, -1, 0), true);
    update(m_psi_e[PSI_EXZ], 2, hfields_n[1], hfields_n[1], IntVect(0, 0, -1), true);
    update(m_psi_e[PSI_EYX], 0, hfields_n[2], hfields_n[2], IntVect(-1, 0, 0), true);
    update(m_psi_e[PSI_EYZ], 2, hfields_n[0], hfields_n[0], IntVect(0, 0, -1), true);
    update(m_psi_e[PSI_EZX], 0, hfields_n[1], hfields_n[1], IntVect(-1, 0, 0), true);
    update(m_psi_e[PSI_EZY], 1, hfields_n[0], hfields_n[0], IntVect(0, -1, 0), true);

    Vector<MultiFab *> psi_e_ptrs{&m_psi_e[PSI_EXY], &m_psi_e[PSI_EXZ], &m_psi_e[PSI_EYX],
                                  &m_psi_e[PSI_EYZ], &m_psi_e[PSI_EZX], &m_psi_e[PSI_EZY]};
    amrex::FillBoundary(psi_e_ptrs, period);
    for (int ip = 0; ip < 6; ++ip)
    {
        m_psi_e[ip].setBndry(0.0_rt);
    }
}

void ADI::updatePmlAuxMagneticFirstHalf(FieldArray const &efields_half,
                                        FieldArray const &efields_n)
{
    auto const period = m_geom.periodicity();
    auto const dxinv = m_geom.InvCellSizeArray();
    auto update = [&](MultiFab &psi_dst, int deriv_dir, MultiFab const &src_hi,
                      MultiFab const &src_lo, IntVect const &off) {
        MultiFab acoef = PmlMakeCoeffLike(psi_dst, 0.0_rt);
        MultiFab bcoef = PmlMakeCoeffLike(psi_dst, 1.0_rt);
        PmlCopyCoefToLayout(acoef, m_a_pml[deriv_dir], period);
        PmlCopyCoefToLayout(bcoef, m_b_pml[deriv_dir], period);
        auto const &psi = psi_dst.arrays();
        auto const &aa = acoef.const_arrays();
        auto const &bb = bcoef.const_arrays();
        auto const &hi = src_hi.const_arrays();
        auto const &lo = src_lo.const_arrays();
        Real const dinv = dxinv[deriv_dir];
        ParallelFor(psi_dst, [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                    {
            int const ii = i + off[0];
            int const jj = j + off[1];
            int const kk = k + off[2];
            // Magnetic-field curls use forward Yee differences.
            Real const dnum = lo[b](ii, jj, kk) - hi[b](i, j, k);
            psi[b](i, j, k) = bb[b](i, j, k) * psi[b](i, j, k) +
                              aa[b](i, j, k) * dinv * dnum; });
    };

    update(m_psi_h[PSI_HXY - PSI_HXY], 1, efields_n[2], efields_n[2], IntVect(0, 1, 0));
    update(m_psi_h[PSI_HXZ - PSI_HXY], 2, efields_half[1], efields_half[1], IntVect(0, 0, 1));
    update(m_psi_h[PSI_HYX - PSI_HXY], 0, efields_half[2], efields_half[2], IntVect(1, 0, 0));
    update(m_psi_h[PSI_HYZ - PSI_HXY], 2, efields_n[0], efields_n[0], IntVect(0, 0, 1));
    update(m_psi_h[PSI_HZX - PSI_HXY], 0, efields_n[1], efields_n[1], IntVect(1, 0, 0));
    update(m_psi_h[PSI_HZY - PSI_HXY], 1, efields_half[0], efields_half[0], IntVect(0, 1, 0));

    Vector<MultiFab *> psi_h_ptrs{&m_psi_h[PSI_HXY - PSI_HXY], &m_psi_h[PSI_HXZ - PSI_HXY],
                                  &m_psi_h[PSI_HYX - PSI_HXY], &m_psi_h[PSI_HYZ - PSI_HXY],
                                  &m_psi_h[PSI_HZX - PSI_HXY], &m_psi_h[PSI_HZY - PSI_HXY]};
    amrex::FillBoundary(psi_h_ptrs, period);
    for (int ip = 0; ip < 6; ++ip)
    {
        m_psi_h[ip].setBndry(0.0_rt);
    }
}

void ADI::updatePmlAuxElectricSecondHalf(FieldArray const &hfields_half)
{
    updatePmlAuxElectricFirstHalf(hfields_half);
}

void ADI::updatePmlAuxMagneticSecondHalf(FieldArray const &efields_np1,
                                         FieldArray const &efields_half)
{
    auto const period = m_geom.periodicity();
    auto const dxinv = m_geom.InvCellSizeArray();
    auto update = [&](MultiFab &psi_dst, int deriv_dir, MultiFab const &src_hi,
                      MultiFab const &src_lo, IntVect const &off) {
        MultiFab acoef = PmlMakeCoeffLike(psi_dst, 0.0_rt);
        MultiFab bcoef = PmlMakeCoeffLike(psi_dst, 1.0_rt);
        PmlCopyCoefToLayout(acoef, m_a_pml[deriv_dir], period);
        PmlCopyCoefToLayout(bcoef, m_b_pml[deriv_dir], period);
        auto const &psi = psi_dst.arrays();
        auto const &aa = acoef.const_arrays();
        auto const &bb = bcoef.const_arrays();
        auto const &hi = src_hi.const_arrays();
        auto const &lo = src_lo.const_arrays();
        Real const dinv = dxinv[deriv_dir];
        ParallelFor(psi_dst, [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                    {
            int const ii = i + off[0];
            int const jj = j + off[1];
            int const kk = k + off[2];
            // Magnetic-field curls use forward Yee differences.
            Real const dnum = lo[b](ii, jj, kk) - hi[b](i, j, k);
            psi[b](i, j, k) = bb[b](i, j, k) * psi[b](i, j, k) +
                              aa[b](i, j, k) * dinv * dnum; });
    };

    update(m_psi_h[PSI_HXY - PSI_HXY], 1, efields_np1[2], efields_np1[2], IntVect(0, 1, 0));
    update(m_psi_h[PSI_HXZ - PSI_HXY], 2, efields_half[1], efields_half[1], IntVect(0, 0, 1));
    update(m_psi_h[PSI_HYX - PSI_HXY], 0, efields_half[2], efields_half[2], IntVect(1, 0, 0));
    update(m_psi_h[PSI_HYZ - PSI_HXY], 2, efields_half[0], efields_half[0], IntVect(0, 0, 1));
    update(m_psi_h[PSI_HZX - PSI_HXY], 0, efields_half[1], efields_half[1], IntVect(1, 0, 0));
    update(m_psi_h[PSI_HZY - PSI_HXY], 1, efields_np1[0], efields_np1[0], IntVect(0, 1, 0));

    Vector<MultiFab *> psi_h_ptrs{&m_psi_h[PSI_HXY - PSI_HXY], &m_psi_h[PSI_HXZ - PSI_HXY],
                                  &m_psi_h[PSI_HYX - PSI_HXY], &m_psi_h[PSI_HYZ - PSI_HXY],
                                  &m_psi_h[PSI_HZX - PSI_HXY], &m_psi_h[PSI_HZY - PSI_HXY]};
    amrex::FillBoundary(psi_h_ptrs, period);
    for (int ip = 0; ip < 6; ++ip)
    {
        m_psi_h[ip].setBndry(0.0_rt);
    }
}

void ADI::stepHxPml1(MultiFab &hx_dst, MultiFab const &ey_half,
                     MultiFab const &ez_n, Real dt)
{
    amrex::ignore_unused(dt);
    auto const period = m_geom.periodicity();
    auto const dxinv = m_geom.InvCellSizeArray();
    MultiFab kz = PmlMakeCoeffLike(hx_dst, 1.0_rt);
    MultiFab ky = PmlMakeCoeffLike(hx_dst, 1.0_rt);
    PmlCopyCoefToLayout(kz, m_kappa[2], period);
    PmlCopyCoefToLayout(ky, m_kappa[1], period);
    auto const &db = m_Db[0].const_arrays();
    auto const &ey = ey_half.const_arrays();
    auto const &ez = ez_n.const_arrays();
    auto const &kz_arr = kz.const_arrays();
    auto const &ky_arr = ky.const_arrays();
    auto const &psi_hxz = m_psi_h[PSI_HXZ - PSI_HXY].const_arrays();
    auto const &psi_hxy = m_psi_h[PSI_HXY - PSI_HXY].const_arrays();
    auto const &hx = hx_dst.arrays();

    ParallelFor(hx_dst, [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                {
        Real const r = db[b](i, j, k);
        hx[b](i, j, k) += r * ((dxinv[2] / kz_arr[b](i, j, k)) *
                                   (ey[b](i, j, k + 1) - ey[b](i, j, k)) -
                               (dxinv[1] / ky_arr[b](i, j, k)) *
                                   (ez[b](i, j + 1, k) - ez[b](i, j, k)) +
                               psi_hxz[b](i, j, k) - psi_hxy[b](i, j, k)); });
}

void ADI::stepHyPml1(MultiFab &hy_dst, MultiFab const &ez_half,
                     MultiFab const &ex_n, Real dt)
{
    amrex::ignore_unused(dt);
    auto const period = m_geom.periodicity();
    auto const dxinv = m_geom.InvCellSizeArray();
    MultiFab kx = PmlMakeCoeffLike(hy_dst, 1.0_rt);
    MultiFab kz = PmlMakeCoeffLike(hy_dst, 1.0_rt);
    PmlCopyCoefToLayout(kx, m_kappa[0], period);
    PmlCopyCoefToLayout(kz, m_kappa[2], period);
    auto const &db = m_Db[1].const_arrays();
    auto const &ez = ez_half.const_arrays();
    auto const &ex = ex_n.const_arrays();
    auto const &kx_arr = kx.const_arrays();
    auto const &kz_arr = kz.const_arrays();
    auto const &psi_hyx = m_psi_h[PSI_HYX - PSI_HXY].const_arrays();
    auto const &psi_hyz = m_psi_h[PSI_HYZ - PSI_HXY].const_arrays();
    auto const &hy = hy_dst.arrays();

    ParallelFor(hy_dst, [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                {
        Real const r = db[b](i, j, k);
        hy[b](i, j, k) += r * ((dxinv[0] / kx_arr[b](i, j, k)) *
                                   (ez[b](i + 1, j, k) - ez[b](i, j, k)) -
                               (dxinv[2] / kz_arr[b](i, j, k)) *
                                   (ex[b](i, j, k + 1) - ex[b](i, j, k)) +
                               psi_hyx[b](i, j, k) - psi_hyz[b](i, j, k)); });
}

void ADI::stepHzPml1(MultiFab &hz_dst, MultiFab const &ex_half,
                     MultiFab const &ey_n, Real dt)
{
    amrex::ignore_unused(dt);
    auto const period = m_geom.periodicity();
    auto const dxinv = m_geom.InvCellSizeArray();
    MultiFab ky = PmlMakeCoeffLike(hz_dst, 1.0_rt);
    MultiFab kx = PmlMakeCoeffLike(hz_dst, 1.0_rt);
    PmlCopyCoefToLayout(ky, m_kappa[1], period);
    PmlCopyCoefToLayout(kx, m_kappa[0], period);
    auto const &db = m_Db[2].const_arrays();
    auto const &ex = ex_half.const_arrays();
    auto const &ey = ey_n.const_arrays();
    auto const &ky_arr = ky.const_arrays();
    auto const &kx_arr = kx.const_arrays();
    auto const &psi_hzy = m_psi_h[PSI_HZY - PSI_HXY].const_arrays();
    auto const &psi_hzx = m_psi_h[PSI_HZX - PSI_HXY].const_arrays();
    auto const &hz = hz_dst.arrays();

    ParallelFor(hz_dst, [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                {
        Real const r = db[b](i, j, k);
        hz[b](i, j, k) += r * ((dxinv[1] / ky_arr[b](i, j, k)) *
                                   (ex[b](i, j + 1, k) - ex[b](i, j, k)) -
                               (dxinv[0] / kx_arr[b](i, j, k)) *
                                   (ey[b](i + 1, j, k) - ey[b](i, j, k)) +
                               psi_hzy[b](i, j, k) - psi_hzx[b](i, j, k)); });
}

void ADI::stepHxPml2(MultiFab &hx_dst, MultiFab const &ey_half,
                     MultiFab const &ez_np1, Real dt)
{
    stepHxPml1(hx_dst, ey_half, ez_np1, dt);
}

void ADI::stepHyPml2(MultiFab &hy_dst, MultiFab const &ez_half,
                     MultiFab const &ex_np1, Real dt)
{
    stepHyPml1(hy_dst, ez_half, ex_np1, dt);
}

void ADI::stepHzPml2(MultiFab &hz_dst, MultiFab const &ex_half,
                     MultiFab const &ey_np1, Real dt)
{
    stepHzPml1(hz_dst, ex_half, ey_np1, dt);
}
