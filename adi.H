#ifndef ADI_H_
#define ADI_H_

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <string>

class ADI
{
public:

    ADI ();

    void initData ();
    void evolve ();

private:
    amrex::IntVect m_n_cells;
    int m_max_grid_size = 64;
    amrex::Geometry m_geom;
    amrex::BoxArray m_grids; // cell-centered grids
    amrex::DistributionMapping m_dmap;

    int m_max_step = 100;
    int m_plot_int = 0;
    amrex::Real m_cfl = 0.99;
    std::string m_plot_format = "numpy";  // "numpy" or "visit"
    std::string m_output_dir = "numpy_out";

    std::string m_ic = "sinwave";
    amrex::Real m_sinwave_amplitude = 1.0;
    int m_sinwave_dir = 2;   // propagation: 0=+x, 1=+y, 2=+z
    int m_sinwave_pol = 0;   // E component: 0=Ex, 1=Ey, 2=Ez
    amrex::Real m_sinwave_wavelength = -1.0; // <0: one wavelength along dir

    amrex::Array<amrex::MultiFab,AMREX_SPACEDIM> m_efields;
    amrex::Array<amrex::MultiFab,AMREX_SPACEDIM> m_bfields;
};

#endif
