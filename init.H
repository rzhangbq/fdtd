#ifndef INIT_H_
#define INIT_H_

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <string>

void InitSetupFields(
    std::string const &pp_prefix,
    std::string const &ic,
    amrex::Real ic_amplitude,
    int ic_dir,
    int ic_pol,
    amrex::Real ic_wavelength,
    amrex::Real pulse_center,
    amrex::Real pulse_sigma,
    amrex::Real eps_r,
    amrex::Real mu_r,
    amrex::Geometry const &geom,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &efields,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &magnetic_fields,
    bool magnetic_fields_are_h = false);

#endif
