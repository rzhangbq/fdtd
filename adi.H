#ifndef ADI_H_
#define ADI_H_

#include <string>

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

class ADI
{
public:
    using FieldArray = amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>;

    ADI();

    void initData();
    void evolve();

    // First ADI half-step (eq:adi-first-half-amrex in notes/adi.tex)
    void adiFirstHalfStep(FieldArray &efields, FieldArray &bfields,
                          FieldArray &efields_x, FieldArray &efields_y,
                          FieldArray &efields_z, FieldArray &bfields_x,
                          FieldArray &bfields_y, FieldArray &bfields_z,
                          amrex::Real dt);
    // Second ADI half-step (eq:adi-second-half-amrex in notes/adi.tex)
    void adiSecondHalfStep(FieldArray &efields, FieldArray &bfields,
                           FieldArray &efields_x, FieldArray &efields_y,
                           FieldArray &efields_z, FieldArray &bfields_x,
                           FieldArray &bfields_y, FieldArray &bfields_z,
                           amrex::Real dt);

    // buildRhs / solveImplicit stubs (to be filled in later)
    amrex::MultiFab buildRhsEx1(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &efields,
                                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &bfields,
                                amrex::Real dt) const;
    amrex::MultiFab buildRhsEy1(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &efields,
                                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &bfields,
                                amrex::Real dt) const;
    amrex::MultiFab buildRhsEz1(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &efields,
                                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &bfields,
                                amrex::Real dt) const;
    void solveImplicitEx1(amrex::MultiFab &ex, amrex::MultiFab const &rhs,
                          amrex::Real dt) const;
    void solveImplicitEy1(amrex::MultiFab &ey, amrex::MultiFab const &rhs,
                          amrex::Real dt) const;
    void solveImplicitEz1(amrex::MultiFab &ez, amrex::MultiFab const &rhs,
                          amrex::Real dt) const;

    amrex::MultiFab buildRhsEx2(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &efields,
                                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &bfields,
                                amrex::Real dt) const;
    amrex::MultiFab buildRhsEy2(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &efields,
                                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &bfields,
                                amrex::Real dt) const;
    amrex::MultiFab buildRhsEz2(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &efields,
                                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &bfields,
                                amrex::Real dt) const;
    void solveImplicitEx2(amrex::MultiFab &ex, amrex::MultiFab const &rhs,
                          amrex::Real dt) const;
    void solveImplicitEy2(amrex::MultiFab &ey, amrex::MultiFab const &rhs,
                          amrex::Real dt) const;
    void solveImplicitEz2(amrex::MultiFab &ez, amrex::MultiFab const &rhs,
                          amrex::Real dt) const;

    // Internal helper updates for H += D_b curl E (magnetic field stored as H).
    void stepBx(amrex::MultiFab &bx, amrex::MultiFab const &ey_src,
                amrex::MultiFab const &ez_src,
                amrex::Real dt);
    void stepBy(amrex::MultiFab &by, amrex::MultiFab const &ez_src,
                amrex::MultiFab const &ex_src,
                amrex::Real dt);
    void stepBz(amrex::MultiFab &bz, amrex::MultiFab const &ex_src,
                amrex::MultiFab const &ey_src,
                amrex::Real dt);

private:
    void initMaterials();
    void updateMaterialCoeffs(amrex::Real dt);
    void convertBtoH(amrex::Real dt);

    // Yee-staggered ADI half-step coefficients. Physical eps/mu/sigma are
    // read from inputs and folded into these fields at initialization/update.
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_Cb;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_p;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_Db;

    amrex::Real m_eps_r = 1.0;
    amrex::Real m_mu_r = 1.0;
    amrex::Real m_sigma_bg = 0.0;
    amrex::RealVect m_block_lo{-1.0, -1.0, -1.0};
    amrex::RealVect m_block_hi{-1.0, -1.0, -1.0};
    amrex::Real m_block_eps_r = 1.0;
    amrex::Real m_block_mu_r = 1.0;
    amrex::Real m_block_sigma = 0.0;
    bool m_has_material_block = false;
    bool m_magnetic_fields_are_h = false;

    amrex::IntVect m_n_cells;
    int m_max_grid_size = 64;
    amrex::Geometry m_geom;
    amrex::BoxArray m_grids; // cell-centered grids
    amrex::DistributionMapping m_dmap;

    int m_max_step = 100;
    int m_plot_int = 0;
    amrex::Real m_cfl = 0.99;  // >0: CFL fraction; <0: use m_dt
    amrex::Real m_dt = -1.0;   // >0: fixed timestep [s]; <0: use m_cfl
    std::string m_plot_format = "numpy"; // "numpy" or "visit"
    std::string m_output_dir = "numpy_out";
    bool m_conv_plt = false; // numpy output: line along ic_dir through domain center

    std::string m_ic = "sinwave";
    amrex::Real m_ic_amplitude = 1.0;
    int m_ic_dir = 2;                   // propagation: 0=+x, 1=+y, 2=+z
    int m_ic_pol = 0;                   // E component: 0=Ex, 1=Ey, 2=Ez
    amrex::Real m_ic_wavelength = -1.0; // sinwave: <0 => one wavelength along ic_dir

    amrex::Real m_pulse_center = 0.5; // xi_0 along ic_dir
    amrex::Real m_pulse_sigma = -1.0; // <0: 0.1 * min domain extent

    // PEC walls normal to this direction (0=x, 1=y, 2=z); -1 = fully periodic.
    int m_pec_normal = -1;
    // 0 = PEC on domain faces normal to pec_normal; else interior PEC at this index.
    int m_pec_location = 0;

    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_efields;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_bfields;
};

#endif
