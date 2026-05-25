
#include "fdtd.H"

#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Utility.H>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>

using namespace amrex;

FDTD::FDTD ()
{
    ParmParse pp("fdtd");
    pp.getarr("n_cells", m_n_cells);
    pp.query("max_grid_size", m_max_grid_size);

    RealVect prob_lo, prob_hi;
    pp.getarr("prob_lo", prob_lo);
    pp.getarr("prob_hi", prob_hi);

    pp.query("max_step", m_max_step);
    pp.query("plot_int", m_plot_int);
    pp.query("cfl", m_cfl);
    pp.query("output_dir", m_output_dir);

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

void FDTD::initData ()
{
    // for now let's just set them to zero
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        m_efields[idim].setVal(0);
        m_bfields[idim].setVal(0);
    }
}

void FDTD::writeNumpyOutput (int step, Real time) const
{
    MultiFab plotmf(m_grids, m_dmap, 6, 0);

    Vector<const MultiFab*> efields{
        AMREX_D_DECL(&m_efields[0], &m_efields[1], &m_efields[2])
    };
    Vector<const MultiFab*> bfields{
        AMREX_D_DECL(&m_bfields[0], &m_bfields[1], &m_bfields[2])
    };

    average_edge_to_cellcenter(plotmf, 0, efields);
    average_face_to_cellcenter(plotmf, 3, bfields);

    const Box& domain_box = m_geom.Domain();
    const auto lo = domain_box.loVect();
    const auto hi = domain_box.hiVect();
    const int nx = hi[0] - lo[0] + 1;
    const int ny = hi[1] - lo[1] + 1;
    const int nz = hi[2] - lo[2] + 1;
    constexpr int ncomp = 6;

    Vector<double> data(static_cast<std::size_t>(nx) * ny * nz * ncomp, 0.0);

    for (MFIter mfi(plotmf); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto const& arr = plotmf.const_array(mfi);
        for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k) {
            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                    const std::size_t base =
                        (static_cast<std::size_t>(i - lo[0]) * ny + (j - lo[1])) * nz
                        + (k - lo[2]);
                    for (int c = 0; c < ncomp; ++c) {
                        data[base * ncomp + c] = static_cast<double>(arr(i, j, k, c));
                    }
                }
            }
        }
    }

    if (ParallelDescriptor::MyProc() != 0) {
        return;
    }

    UtilCreateDirectory(m_output_dir, 0755);

    std::ostringstream step_tag;
    step_tag << std::setw(5) << std::setfill('0') << step;

    const std::string bin_file = m_output_dir + "/step_" + step_tag.str() + "_fields.bin";
    const std::string meta_file = m_output_dir + "/step_" + step_tag.str() + "_meta.json";

    {
        std::ofstream ofs(bin_file, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(data.data()),
                  static_cast<std::streamsize>(data.size() * sizeof(double)));
    }

    auto problo = m_geom.ProbLoArray();
    auto probhi = m_geom.ProbHiArray();

    std::ofstream meta(meta_file);
    meta << std::setprecision(17);
    meta << "{\n";
    meta << "  \"step\": " << step << ",\n";
    meta << "  \"time\": " << time << ",\n";
    meta << "  \"shape\": [" << nx << ", " << ny << ", " << nz << ", " << ncomp << "],\n";
    meta << "  \"dtype\": \"float64\",\n";
    meta << "  \"layout\": \"C\",\n";
    meta << "  \"components\": [\"Ex\", \"Ey\", \"Ez\", \"Bx\", \"By\", \"Bz\"],\n";
    meta << "  \"fields_file\": \"" << bin_file << "\",\n";
    meta << "  \"prob_lo\": [" << problo[0] << ", " << problo[1] << ", " << problo[2] << "],\n";
    meta << "  \"prob_hi\": [" << probhi[0] << ", " << probhi[1] << ", " << probhi[2] << "]\n";
    meta << "}\n";
}

void FDTD::evolve ()
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
        writeNumpyOutput(0, time);
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
            writeNumpyOutput(step + 1, time);
        }
    }
}
