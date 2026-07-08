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
    void adiFirstHalfStep(FieldArray &efields, FieldArray &hfields,
                          FieldArray &efields_x, FieldArray &efields_y,
                          FieldArray &efields_z, FieldArray &hfields_x,
                          FieldArray &hfields_y, FieldArray &hfields_z,
                          amrex::Real dt);
    // Second ADI half-step (eq:adi-second-half-amrex in notes/adi.tex)
    void adiSecondHalfStep(FieldArray &efields, FieldArray &hfields,
                           FieldArray &efields_x, FieldArray &efields_y,
                           FieldArray &efields_z, FieldArray &hfields_x,
                           FieldArray &hfields_y, FieldArray &hfields_z,
                           amrex::Real dt);

    // buildRhs / solveImplicit stubs (to be filled in later)
    amrex::MultiFab buildRhsEx1(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &efields,
                                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &hfields,
                                amrex::Real dt) const;
    amrex::MultiFab buildRhsEy1(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &efields,
                                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &hfields,
                                amrex::Real dt) const;
    amrex::MultiFab buildRhsEz1(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &efields,
                                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &hfields,
                                amrex::Real dt) const;
    void solveImplicitEx1(amrex::MultiFab &ex, amrex::MultiFab const &rhs,
                          amrex::Real dt) const;
    void solveImplicitEy1(amrex::MultiFab &ey, amrex::MultiFab const &rhs,
                          amrex::Real dt) const;
    void solveImplicitEz1(amrex::MultiFab &ez, amrex::MultiFab const &rhs,
                          amrex::Real dt) const;

    amrex::MultiFab buildRhsEx2(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &efields,
                                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &hfields,
                                amrex::Real dt) const;
    amrex::MultiFab buildRhsEy2(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &efields,
                                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &hfields,
                                amrex::Real dt) const;
    amrex::MultiFab buildRhsEz2(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &efields,
                                amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &hfields,
                                amrex::Real dt) const;
    void solveImplicitEx2(amrex::MultiFab &ex, amrex::MultiFab const &rhs,
                          amrex::Real dt) const;
    void solveImplicitEy2(amrex::MultiFab &ey, amrex::MultiFab const &rhs,
                          amrex::Real dt) const;
    void solveImplicitEz2(amrex::MultiFab &ez, amrex::MultiFab const &rhs,
                          amrex::Real dt) const;

    // Internal helper updates for H += D_b curl E (magnetic field stored as H).
    void stepHx(amrex::MultiFab &hx, amrex::MultiFab const &ey_src,
                amrex::MultiFab const &ez_src,
                amrex::Real dt);
    void stepHy(amrex::MultiFab &hy, amrex::MultiFab const &ez_src,
                amrex::MultiFab const &ex_src,
                amrex::Real dt);
    void stepHz(amrex::MultiFab &hz, amrex::MultiFab const &ex_src,
                amrex::MultiFab const &ey_src,
                amrex::Real dt);
    void stepHxPml1(amrex::MultiFab &hx, amrex::MultiFab const &ey_half,
                    amrex::MultiFab const &ez_n, amrex::Real dt);
    void stepHyPml1(amrex::MultiFab &hy, amrex::MultiFab const &ez_half,
                    amrex::MultiFab const &ex_n, amrex::Real dt);
    void stepHzPml1(amrex::MultiFab &hz, amrex::MultiFab const &ex_half,
                    amrex::MultiFab const &ey_n, amrex::Real dt);
    void stepHxPml2(amrex::MultiFab &hx, amrex::MultiFab const &ey_half,
                    amrex::MultiFab const &ez_np1, amrex::Real dt);
    void stepHyPml2(amrex::MultiFab &hy, amrex::MultiFab const &ez_half,
                    amrex::MultiFab const &ex_np1, amrex::Real dt);
    void stepHzPml2(amrex::MultiFab &hz, amrex::MultiFab const &ex_half,
                    amrex::MultiFab const &ey_np1, amrex::Real dt);

private:
    void initMaterials();
    void updateMaterialCoeffs(amrex::Real dt);
    void initPmlProfiles();
    void updatePmlRecursionCoeffs(amrex::Real dt);
    void updatePmlAuxElectricFirstHalf(FieldArray const &hfields_n);
    void updatePmlAuxMagneticFirstHalf(FieldArray const &efields_half,
                                       FieldArray const &efields_n);
    void updatePmlAuxElectricSecondHalf(FieldArray const &hfields_half);
    void updatePmlAuxMagneticSecondHalf(FieldArray const &efields_np1,
                                        FieldArray const &efields_half);

    // Yee-staggered ADI half-step coefficients. Physical eps/mu/sigma are
    // read from inputs and folded into these fields at initialization/update.
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_Cb;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_p;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_Db;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_kappa;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_a_pml;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_b_pml;
    enum PsiIndex
    {
        PSI_EXY = 0,
        PSI_EXZ,
        PSI_EYX,
        PSI_EYZ,
        PSI_EZX,
        PSI_EZY,
        PSI_HXY,
        PSI_HXZ,
        PSI_HYX,
        PSI_HYZ,
        PSI_HZX,
        PSI_HZY
    };
    amrex::Array<amrex::MultiFab, 6> m_psi_e;
    amrex::Array<amrex::MultiFab, 6> m_psi_h;

    amrex::Real m_eps_r = 1.0;
    amrex::Real m_mu_r = 1.0;
    amrex::Real m_sigma_bg = 0.0;
    amrex::RealVect m_block_lo{-1.0, -1.0, -1.0};
    amrex::RealVect m_block_hi{-1.0, -1.0, -1.0};
    amrex::Real m_block_eps_r = 1.0;
    amrex::Real m_block_mu_r = 1.0;
    amrex::Real m_block_sigma = 0.0;
    bool m_has_material_block = false;

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
    bool m_pml_on = false;
    int m_pml_normal = -1;
    int m_pml_thickness = 10;
    amrex::Real m_pml_sigma_max = 0.0;
    amrex::Real m_pml_grade_m = 3.0;
    amrex::Real m_pml_alpha_max = 0.0;
    amrex::Real m_pml_kappa_max = 1.0;

    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_efields;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_hfields;
};

#endif
