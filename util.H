#ifndef UTIL_H_
#define UTIL_H_

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <string>

void UtilBuildCellCenteredPlotMF(
    amrex::MultiFab &plotmf,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &efields,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &bfields);

void UtilWritePlotOutput(
    std::string const &plot_format,
    std::string const &output_dir,
    int step,
    amrex::Real time,
    amrex::BoxArray const &grids,
    amrex::DistributionMapping const &dmap,
    amrex::Geometry const &geom,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &efields,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &bfields);

// Zero tangential E on PEC planes. pec_normal: wall normal (0=x, 1=y, 2=z); -1 = no-op.
void UtilEnforcePecEfields(
    int pec_normal,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &efields);

#endif
