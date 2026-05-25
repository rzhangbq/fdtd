#ifndef FDTD_H_
#define FDTD_H_

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <string>

class FDTD
{
public:

    FDTD ();

    void initData ();
    void evolve ();

private:
    void writeNumpyOutput (int step, amrex::Real time) const;

    amrex::IntVect m_n_cells;
    int m_max_grid_size = 64;
    amrex::Geometry m_geom;
    amrex::BoxArray m_grids; // cell-centered grids
    amrex::DistributionMapping m_dmap;

    int m_max_step = 100;
    int m_plot_int = 0;
    amrex::Real m_cfl = 0.99;
    std::string m_output_dir = "numpy_out";

    amrex::Array<amrex::MultiFab,AMREX_SPACEDIM> m_efields;
    amrex::Array<amrex::MultiFab,AMREX_SPACEDIM> m_bfields;
};

#endif
