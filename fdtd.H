#ifndef FDTD_H_
#define FDTD_H_

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <string>

class FDTD
{
public:
    FDTD();

    void initData();
    void evolve();

private:
    amrex::IntVect m_n_cells;
    int m_max_grid_size = 64;
    amrex::Geometry m_geom;
    amrex::BoxArray m_grids; // cell-centered grids
    amrex::DistributionMapping m_dmap;

    int m_max_step = 100;
    int m_plot_int = 0;
    amrex::Real m_cfl = 0.99;
    std::string m_plot_format = "numpy"; // "numpy" or "visit"
    std::string m_output_dir = "numpy_out";

    std::string m_ic = "sinwave";
    amrex::Real m_ic_amplitude = 1.0;
    int m_ic_dir = 2;                   // propagation: 0=+x, 1=+y, 2=+z
    int m_ic_pol = 0;                   // E component: 0=Ex, 1=Ey, 2=Ez
    amrex::Real m_ic_wavelength = -1.0; // sinwave: <0 => one wavelength along ic_dir

    amrex::Real m_pulse_center = 0.5; // xi_0 along ic_dir
    amrex::Real m_pulse_sigma = -1.0; // <0: 0.1 * min domain extent

    // PEC walls normal to this direction (0=x, 1=y, 2=z); -1 = fully periodic.
    int m_pec_normal = -1;

    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_efields;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_bfields;
};

#endif
