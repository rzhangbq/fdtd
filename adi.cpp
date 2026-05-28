
#include "adi.H"
#include "util.H"

#include <AMReX_ParmParse.H>

#include <cmath>

using namespace amrex;

namespace {
constexpr Real c_light = 2.99792458e8;

// Return the physical coordinate in one direction for a field component that is
// staggered on a Yee grid: half-cell shifted along its own component direction.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real staggered_coord (int component, int coord_dir, int i, int j, int k,
                      GpuArray<Real,AMREX_SPACEDIM> const& problo,
                      GpuArray<Real,AMREX_SPACEDIM> const& dx)
{
    int idx = (coord_dir == 0) ? i : ((coord_dir == 1) ? j : k);
    Real offset = (component == coord_dir) ? 0.5_rt : 0.0_rt;
    return problo[coord_dir] + (idx + offset) * dx[coord_dir];
}
} // namespace

ADI::ADI ()
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

    if (m_plot_format != "numpy" && m_plot_format != "visit") {
        amrex::Abort("adi.plot_format must be \"numpy\" or \"visit\"");
    }

    pp.query("ic", m_ic);
    pp.query("sinwave_amplitude", m_sinwave_amplitude);
    pp.query("sinwave_dir", m_sinwave_dir);
    pp.query("sinwave_pol", m_sinwave_pol);
    pp.query("sinwave_wavelength", m_sinwave_wavelength);

    Box domain(IntVect(0), m_n_cells-1);
    RealBox real_box(prob_lo.begin(), prob_hi.begin());
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1,1,1)};

    m_geom.define(domain, real_box, CoordSys::cartesian, is_periodic);

    m_grids.define(domain);
    m_grids.maxSize(m_max_grid_size);

    m_dmap.define(m_grids);

    static_assert(AMREX_SPACEDIM == 3, "3D only");
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        IntVect etyp(1); // nodal by default
        etyp[idim] = 0;  // cell-centered in idim-direction
        m_efields[idim].define(amrex::convert(m_grids,etyp), m_dmap,
                               1, 1); // one component, one ghost
        IntVect btyp(0); // cell-centerd by default
        btyp[idim] = 1;  // nodal in idim-direction
        m_bfields[idim].define(amrex::convert(m_grids,btyp), m_dmap, 1, 1);
    }
}

void ADI::initData ()
{
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        m_efields[idim].setVal(0);
        m_bfields[idim].setVal(0);
    }

    const bool is_sinwave = (m_ic == "sinwave");
    const bool is_standing_wave = (m_ic == "standingwave");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(is_sinwave || is_standing_wave,
                                     "adi.ic must be 'sinwave' or 'standingwave'");

    AMREX_ALWAYS_ASSERT(m_sinwave_dir >= 0 && m_sinwave_dir < AMREX_SPACEDIM);
    AMREX_ALWAYS_ASSERT(m_sinwave_pol >= 0 && m_sinwave_pol < AMREX_SPACEDIM);
    AMREX_ALWAYS_ASSERT(m_sinwave_pol != m_sinwave_dir);

    auto problo = m_geom.ProbLoArray();
    auto probhi = m_geom.ProbHiArray();
    auto dx = m_geom.CellSizeArray();

    Real domain_len = probhi[m_sinwave_dir] - problo[m_sinwave_dir];
    Real wavelength = (m_sinwave_wavelength > 0) ? m_sinwave_wavelength : domain_len;
    Real kw = 2.0 * M_PI / wavelength;
    Real E0 = m_sinwave_amplitude;
    const int dir = m_sinwave_dir;
    const int pol = m_sinwave_pol;

    auto const& ea = m_efields[pol].arrays();
    // determine the direction of the magnetic field and its sign based on the right-hand rule
    const int bdir = 3 - dir - pol;
    const Real bsign = ((dir + 1) % 3 == pol) ? 1.0_rt : -1.0_rt;
    auto const& ba = m_bfields[bdir].arrays();

    ParallelFor(m_efields[pol], [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
    {
        Real phase_coord = staggered_coord(pol, dir, i, j, k, problo, dx);
        Real s = std::sin(kw * phase_coord);
        ea[b](i,j,k) = E0 * s;
    });

    if (is_sinwave) {
        Real B0 = E0 / c_light;
        // B = (1/c) k_hat x E for a +dir traveling wave at t=0
        ParallelFor(m_bfields[bdir], [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
        {
            Real phase_coord = staggered_coord(bdir, dir, i, j, k, problo, dx);
            Real s = std::sin(kw * phase_coord);
            ba[b](i,j,k) = bsign * B0 * s;
        });
    }
    // For a standing wave at t=0, initialize B to zero and only seed E.
    Gpu::streamSynchronize();

    Vector<MultiFab*> efields{AMREX_D_DECL(&m_efields[0], &m_efields[1], &m_efields[2])};
    Vector<MultiFab*> bfields{AMREX_D_DECL(&m_bfields[0], &m_bfields[1], &m_bfields[2])};
    amrex::FillBoundary(efields, m_geom.periodicity());
    amrex::FillBoundary(bfields, m_geom.periodicity());
}

void ADI::evolve ()
{
    constexpr Real c = 2.99792458e8;

    auto dx = m_geom.CellSizeArray();
    Real dt = std::min({dx[0],dx[1],dx[2]}) / c * m_cfl;
    Real c2dt = c*c*dt;
    auto dxinv = m_geom.InvCellSizeArray();

    auto const period = m_geom.periodicity();
    Vector<MultiFab*> efields{AMREX_D_DECL(&m_efields[0], &m_efields[1], &m_efields[2])};
    Vector<MultiFab*> bfields{AMREX_D_DECL(&m_bfields[0], &m_bfields[1], &m_bfields[2])};

    Real time = 0.0_rt;

    if (m_plot_int > 0) {
        UtilWritePlotOutput(m_plot_format, m_output_dir, 0, time,
                            m_grids, m_dmap, m_geom, m_efields, m_bfields);
    }

    for (int step = 0; step < m_max_step; ++step)
    {
        amrex::FillBoundary(efields, period);

        auto const& bx = m_bfields[0].arrays();
        auto const& by = m_bfields[1].arrays();
        auto const& bz = m_bfields[2].arrays();
        auto const& ex = m_efields[0].arrays();
        auto const& ey = m_efields[1].arrays();
        auto const& ez = m_efields[2].arrays();

        Real halfdt = 0.5_rt*dt;
        ParallelFor(m_bfields[0], [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
        {
            bx[b](i,j,k) -= halfdt * (dxinv[1]*(ez[b](i,j+1,k) - ez[b](i,j,k))
                                    - dxinv[2]*(ey[b](i,j,k+1) - ey[b](i,j,k)));
        });
        ParallelFor(m_bfields[1], [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
        {
            by[b](i,j,k) -= halfdt * (dxinv[2]*(ex[b](i,j,k+1) - ex[b](i,j,k))
                                    - dxinv[0]*(ez[b](i+1,j,k) - ez[b](i,j,k)));
        });
        ParallelFor(m_bfields[2], [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
        {
            bz[b](i,j,k) -= halfdt * (dxinv[0]*(ey[b](i+1,j,k) - ey[b](i,j,k))
                                    - dxinv[1]*(ex[b](i,j+1,k) - ex[b](i,j,k)));
        });
        Gpu::streamSynchronize();

        amrex::FillBoundary(bfields, period);

        ParallelFor(m_efields[0], [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
        {
            ex[b](i,j,k) += c2dt * (dxinv[1]*(bz[b](i,j,k) - bz[b](i,j-1,k))
                                  - dxinv[2]*(by[b](i,j,k) - by[b](i,j,k-1)));
        });
        ParallelFor(m_efields[1], [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
        {
            ey[b](i,j,k) += c2dt * (dxinv[2]*(bx[b](i,j,k) - bx[b](i,j,k-1))
                                  - dxinv[0]*(bz[b](i,j,k) - bz[b](i-1,j,k)));
        });
        ParallelFor(m_efields[2], [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
        {
            ez[b](i,j,k) += c2dt * (dxinv[0]*(by[b](i,j,k) - by[b](i-1,j,k))
                                  - dxinv[1]*(bx[b](i,j,k) - bx[b](i,j-1,k)));
        });
        Gpu::streamSynchronize();

        amrex::FillBoundary(efields, period);

        ParallelFor(m_bfields[0], [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
        {
            bx[b](i,j,k) -= halfdt * (dxinv[1]*(ez[b](i,j+1,k) - ez[b](i,j,k))
                                    - dxinv[2]*(ey[b](i,j,k+1) - ey[b](i,j,k)));
        });
        ParallelFor(m_bfields[1], [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
        {
            by[b](i,j,k) -= halfdt * (dxinv[2]*(ex[b](i,j,k+1) - ex[b](i,j,k))
                                    - dxinv[0]*(ez[b](i+1,j,k) - ez[b](i,j,k)));
        });
        ParallelFor(m_bfields[2], [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
        {
            bz[b](i,j,k) -= halfdt * (dxinv[0]*(ey[b](i+1,j,k) - ey[b](i,j,k))
                                    - dxinv[1]*(ex[b](i,j+1,k) - ex[b](i,j,k)));
        });
        Gpu::streamSynchronize();

        time += dt;

        if (m_plot_int > 0 && (step + 1) % m_plot_int == 0) {
            UtilWritePlotOutput(m_plot_format, m_output_dir, step + 1, time,
                                m_grids, m_dmap, m_geom, m_efields, m_bfields);
        }
    }
}
