#include "init.H"

#include <AMReX_Utility.H>

#include <cmath>

using namespace amrex;

namespace
{
    constexpr Real c_light = 2.99792458e8;

    // Return the physical coordinate in one direction for a field component that is
    // staggered on a Yee grid: half-cell shifted along its own component direction.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real staggered_coord(int component, int coord_dir, int i, int j, int k,
                         GpuArray<Real, AMREX_SPACEDIM> const &problo,
                         GpuArray<int, AMREX_SPACEDIM> const &ncells,
                         GpuArray<Real, AMREX_SPACEDIM> const &dx)
    {
        int idx = (coord_dir == 0) ? i : ((coord_dir == 1) ? j : k);
        bool const is_nodal = (component != coord_dir);
        Real offset = is_nodal ? 0.0_rt : 0.5_rt;

        // Periodic nodal endpoints are duplicate valid points; initialize both
        // representations from the same index-space coordinate.
        if (is_nodal)
        {
            if (idx >= ncells[coord_dir])
            {
                idx -= ncells[coord_dir];
            }
        }

        Real coord = problo[coord_dir] + (idx + offset) * dx[coord_dir];
        return coord;
    }

    // Gaussian-modulated plane wave along ic_dir: E0 * exp(-(xi-x0)^2/(2 sigma^2)) * cos(k0*(xi-x0))
    // with xi the staggered coordinate along dir for the given field component.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real gaussian_plane_wave(int component, int dir, int i, int j, int k,
                             GpuArray<Real, AMREX_SPACEDIM> const &problo,
                             GpuArray<int, AMREX_SPACEDIM> const &ncells,
                             GpuArray<Real, AMREX_SPACEDIM> const &dx,
                             Real x0, Real sigma, Real k0)
    {
        Real xi = staggered_coord(component, dir, i, j, k, problo, ncells, dx);
        Real dxi = xi - x0;
        Real envelope = std::exp(-dxi * dxi / (2.0_rt * sigma * sigma));
        Real carrier = std::cos(k0 * dxi);
        return envelope * carrier;
    }
} // namespace

void InitSetupFields(
    std::string const &pp_prefix,
    std::string const &ic,
    Real ic_amplitude,
    int ic_dir,
    int ic_pol,
    Real ic_wavelength,
    Real pulse_center,
    Real pulse_sigma,
    Real eps_r,
    Real mu_r,
    Geometry const &geom,
    Array<MultiFab, AMREX_SPACEDIM> &efields,
    Array<MultiFab, AMREX_SPACEDIM> &bfields)
{
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        efields[idim].setVal(0);
        bfields[idim].setVal(0);
    }

    const bool is_sinwave = (ic == "sinwave");
    const bool is_standing_wave = (ic == "standingwave");
    const bool is_gaussian_pulse = (ic == "gaussianpulse");
    const std::string ic_err = pp_prefix +
                               ".ic must be 'sinwave', 'standingwave', or 'gaussianpulse'";
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        is_sinwave || is_standing_wave || is_gaussian_pulse, ic_err.c_str());

    AMREX_ALWAYS_ASSERT(ic_dir >= 0 && ic_dir < AMREX_SPACEDIM);
    AMREX_ALWAYS_ASSERT(ic_pol >= 0 && ic_pol < AMREX_SPACEDIM);
    AMREX_ALWAYS_ASSERT(ic_pol != ic_dir);

    auto problo = geom.ProbLoArray();
    auto probhi = geom.ProbHiArray();
    auto dx = geom.CellSizeArray();
    GpuArray<int, AMREX_SPACEDIM> const ncells{
        AMREX_D_DECL(geom.Domain().length(0), geom.Domain().length(1), geom.Domain().length(2))};

    Real E0 = ic_amplitude;
    Real const B0 = E0 * std::sqrt(eps_r * mu_r) / c_light;
    const int dir = ic_dir;
    const int pol = ic_pol;
    const int bdir = 3 - dir - pol;
    const Real bsign = ((dir + 1) % 3 == pol) ? 1.0_rt : -1.0_rt;

    Real const domain_len = probhi[dir] - problo[dir];
    Real const wavelength = (ic_wavelength > 0) ? ic_wavelength : domain_len;
    Real const k0 = 2.0 * M_PI / wavelength;

    if (is_gaussian_pulse)
    {
        Real sigma = pulse_sigma;
        if (sigma <= 0.0_rt)
        {
            sigma = 0.1_rt * domain_len;
        }

        Real const x0 = pulse_center;
        auto const &ea = efields[pol].arrays();
        auto const &ba = bfields[bdir].arrays();

        ParallelFor(efields[pol], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                    {
            Real g = gaussian_plane_wave(pol, dir, i, j, k, problo, ncells, dx, x0, sigma, k0);
            ea[b](i, j, k) = E0 * g; });

        // Sample the matched B field along ic_dir at B_bdir Yee locations.
        ParallelFor(bfields[bdir], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                    {
            Real g = gaussian_plane_wave(bdir, dir, i, j, k, problo, ncells, dx, x0, sigma, k0);
            ba[b](i, j, k) = bsign * B0 * g; });
    }
    else
    {
        Real const kw = k0;

        auto const &ea = efields[pol].arrays();
        auto const &ba = bfields[bdir].arrays();

        ParallelFor(efields[pol], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                    {
            Real phase_coord = staggered_coord(pol, dir, i, j, k, problo, ncells, dx);
            Real s = std::sin(kw * phase_coord);
            ea[b](i, j, k) = E0 * s; });

        if (is_sinwave)
        {
            // B = sqrt(mu*eps) k_hat x E for a +dir traveling wave at t=0.
            ParallelFor(bfields[bdir], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k)
                        {
                Real phase_coord = staggered_coord(bdir, dir, i, j, k, problo, ncells, dx);
                Real s = std::sin(kw * phase_coord);
                ba[b](i, j, k) = bsign * B0 * s; });
        }
        // For a standing wave at t=0, initialize B to zero and only seed E.
    }

    Vector<MultiFab *> efield_ptrs{AMREX_D_DECL(&efields[0], &efields[1], &efields[2])};
    Vector<MultiFab *> bfield_ptrs{AMREX_D_DECL(&bfields[0], &bfields[1], &bfields[2])};
    amrex::FillBoundary(efield_ptrs, geom.periodicity());
    amrex::FillBoundary(bfield_ptrs, geom.periodicity());
}
