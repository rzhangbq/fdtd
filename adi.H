#ifndef ADI_H_
#define ADI_H_

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <string>

class ADI
{
public:
    using FieldArray = amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>;

    ADI();

    void initData();
    void evolve();

    // First ADI half-step (eq:adi-first-half-amrex in notes/adi.tex)
    void adiFirstHalfStep(FieldArray& efields, FieldArray& bfields,
                          FieldArray& efields_x, FieldArray& efields_y,
                          FieldArray& efields_z, FieldArray& bfields_x,
                          FieldArray& bfields_y, FieldArray& bfields_z,
                          amrex::Real dt);
    // Second ADI half-step (eq:adi-second-half-amrex in notes/adi.tex)
    void adiSecondHalfStep(FieldArray& efields, FieldArray& bfields,
                           FieldArray& efields_x, FieldArray& efields_y,
                           FieldArray& efields_z, FieldArray& bfields_x,
                           FieldArray& bfields_y, FieldArray& bfields_z,
                           amrex::Real dt);

    // buildRhs / solveImplicit stubs (to be filled in later)
    amrex::MultiFab buildRhsEx1(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const& efields,
                                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const& bfields,
                                amrex::Real dt) const;
    amrex::MultiFab buildRhsEy1(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const& efields,
                                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const& bfields,
                                amrex::Real dt) const;
    amrex::MultiFab buildRhsEz1(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const& efields,
                                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const& bfields,
                                amrex::Real dt) const;
    void solveImplicitEx1(amrex::MultiFab &ex, amrex::MultiFab const &rhs,
                          amrex::Real dt) const;
    void solveImplicitEy1(amrex::MultiFab &ey, amrex::MultiFab const &rhs,
                          amrex::Real dt) const;
    void solveImplicitEz1(amrex::MultiFab &ez, amrex::MultiFab const &rhs,
                          amrex::Real dt) const;

    amrex::MultiFab buildRhsEx2(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const& efields,
                                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const& bfields,
                                amrex::Real dt) const;
    amrex::MultiFab buildRhsEy2(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const& efields,
                                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const& bfields,
                                amrex::Real dt) const;
    amrex::MultiFab buildRhsEz2(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const& efields,
                                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const& bfields,
                                amrex::Real dt) const;
    void solveImplicitEx2(amrex::MultiFab &ex, amrex::MultiFab const &rhs,
                          amrex::Real dt) const;
    void solveImplicitEy2(amrex::MultiFab &ey, amrex::MultiFab const &rhs,
                          amrex::Real dt) const;
    void solveImplicitEz2(amrex::MultiFab &ez, amrex::MultiFab const &rhs,
                          amrex::Real dt) const;

private:
    // Explicit B += (dt/2) curl E with the old/new split used by the ADI half-step.
    void stepBx(amrex::MultiFab &bx, amrex::MultiFab const &ey_src,
                amrex::MultiFab const &ez_src,
                amrex::Real dt);
    void stepBy(amrex::MultiFab &by, amrex::MultiFab const &ez_src,
                amrex::MultiFab const &ex_src,
                amrex::Real dt);
    void stepBz(amrex::MultiFab &bz, amrex::MultiFab const &ex_src,
                amrex::MultiFab const &ey_src,
                amrex::Real dt);

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

    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_efields;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_bfields;
};

#endif
