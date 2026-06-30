#include "util.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Utility.H>

#include <fstream>
#include <iomanip>
#include <cmath>
#include <sstream>

using namespace amrex;

namespace
{
    constexpr Real mu0 = 4.0 * M_PI * 1e-7;

    void writeVisitOutput(
        std::string const &output_dir,
        int step,
        Real time,
        BoxArray const &grids,
        DistributionMapping const &dmap,
        Geometry const &geom,
        Array<MultiFab, AMREX_SPACEDIM> const &efields,
        Array<MultiFab, AMREX_SPACEDIM> const &magnetic_fields,
        bool magnetic_input_is_h)
    {
        MultiFab plotmf(grids, dmap, 6, 0);
        UtilBuildCellCenteredPlotMF(plotmf, efields, magnetic_fields, magnetic_input_is_h);

        const Vector<std::string> varnames = {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"};
        const std::string plotfile = Concatenate(output_dir, step);

        if (ParallelDescriptor::MyProc() == 0)
        {
            amrex::Print() << "Writing plotfile " << plotfile << " at time " << time << "\n";
        }

        WriteSingleLevelPlotfile(plotfile, plotmf, varnames, geom, time, step);
    }

    void writeNumpyOutput(
        std::string const &output_dir,
        int step,
        Real time,
        bool conv_plt,
        int ic_dir,
        BoxArray const &grids,
        DistributionMapping const &dmap,
        Geometry const &geom,
        Array<MultiFab, AMREX_SPACEDIM> const &efields,
        Array<MultiFab, AMREX_SPACEDIM> const &magnetic_fields,
        bool magnetic_input_is_h)
    {
        MultiFab plotmf(grids, dmap, 6, 0);
        UtilBuildCellCenteredPlotMF(plotmf, efields, magnetic_fields, magnetic_input_is_h);

        const Box &domain_box = geom.Domain();
        IntVect out_lo = domain_box.smallEnd();
        IntVect out_hi = domain_box.bigEnd();
        if (conv_plt)
        {
            AMREX_ALWAYS_ASSERT(ic_dir >= 0 && ic_dir < AMREX_SPACEDIM);
            const int mid = domain_box.smallEnd(ic_dir) + domain_box.length(ic_dir) / 2;
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                if (dir == ic_dir)
                {
                    continue;
                }
                out_lo[dir] = mid;
                out_hi[dir] = mid;
            }
        }

        const auto lo = out_lo;
        const auto hi = out_hi;
        const int nx = hi[0] - lo[0] + 1;
        const int ny = hi[1] - lo[1] + 1;
        const int nz = hi[2] - lo[2] + 1;
        constexpr int ncomp = 6;

        std::size_t const nvals = static_cast<std::size_t>(nx) * ny * nz * ncomp;
        Gpu::DeviceVector<double> d_data(nvals);
        double *data_d = d_data.data();

        for (MFIter mfi(plotmf); mfi.isValid(); ++mfi)
        {
            Box bx = mfi.validbox();
            if (conv_plt)
            {
                bx &= Box(out_lo, out_hi);
            }
            auto const &arr = plotmf.const_array(mfi);
            if (bx.ok())
            {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    const std::size_t base =
                        (static_cast<std::size_t>(i - lo[0]) * ny + (j - lo[1])) * nz + (k - lo[2]);
                    for (int c = 0; c < ncomp; ++c)
                    {
                        data_d[base * ncomp + c] = static_cast<double>(arr(i, j, k, c));
                    }
                });
            }
        }

        Vector<double> data(nvals);
        Gpu::copy(Gpu::deviceToHost, d_data.begin(), d_data.end(), data.begin());

        if (ParallelDescriptor::MyProc() != 0)
        {
            return;
        }

        UtilCreateDirectory(output_dir, 0755);

        std::ostringstream step_tag;
        step_tag << std::setw(5) << std::setfill('0') << step;

        const std::string bin_file = output_dir + "/step_" + step_tag.str() + "_fields.bin";
        const std::string meta_file = output_dir + "/step_" + step_tag.str() + "_meta.json";

        {
            std::ofstream ofs(bin_file, std::ios::binary);
            ofs.write(reinterpret_cast<const char *>(data.data()),
                      static_cast<std::streamsize>(data.size() * sizeof(double)));
        }

        auto problo = geom.ProbLoArray();
        auto probhi = geom.ProbHiArray();

        std::ofstream meta(meta_file);
        meta << std::setprecision(17);
        meta << "{\n";
        meta << "  \"step\": " << step << ",\n";
        meta << "  \"time\": " << time << ",\n";
        meta << "  \"shape\": [" << nx << ", " << ny << ", " << nz << ", " << ncomp << "],\n";
        meta << "  \"dtype\": \"float64\",\n";
        meta << "  \"layout\": \"C\",\n";
        meta << "  \"components\": [\"Ex\", \"Ey\", \"Ez\", \"Hx\", \"Hy\", \"Hz\"],\n";
        meta << "  \"conv_plt\": " << (conv_plt ? "true" : "false") << ",\n";
        meta << "  \"line_axis\": " << ic_dir << ",\n";
        meta << "  \"fields_file\": \"" << bin_file << "\",\n";
        meta << "  \"prob_lo\": [" << problo[0] << ", " << problo[1] << ", " << problo[2] << "],\n";
        meta << "  \"prob_hi\": [" << probhi[0] << ", " << probhi[1] << ", " << probhi[2] << "]\n";
        meta << "}\n";
    }
} // namespace

void UtilBuildCellCenteredPlotMF(
    MultiFab &plotmf,
    Array<MultiFab, AMREX_SPACEDIM> const &efields,
    Array<MultiFab, AMREX_SPACEDIM> const &magnetic_fields,
    bool magnetic_input_is_h)
{
    Vector<const MultiFab *> efield_ptrs{
        AMREX_D_DECL(&efields[0], &efields[1], &efields[2])};
    Vector<const MultiFab *> magnetic_field_ptrs{
        AMREX_D_DECL(&magnetic_fields[0], &magnetic_fields[1], &magnetic_fields[2])};

    average_edge_to_cellcenter(plotmf, 0, efield_ptrs);
    average_face_to_cellcenter(plotmf, 3, magnetic_field_ptrs);
    if (!magnetic_input_is_h)
    {
        plotmf.mult(1.0_rt / mu0, 3, AMREX_SPACEDIM, 0);
    }
}

void UtilWritePlotOutput(
    std::string const &plot_format,
    std::string const &output_dir,
    int step,
    Real time,
    bool conv_plt,
    int ic_dir,
    BoxArray const &grids,
    DistributionMapping const &dmap,
    Geometry const &geom,
    Array<MultiFab, AMREX_SPACEDIM> const &efields,
    Array<MultiFab, AMREX_SPACEDIM> const &magnetic_fields,
    bool magnetic_input_is_h)
{
    if (plot_format == "visit")
    {
        writeVisitOutput(output_dir, step, time, grids, dmap, geom,
                         efields, magnetic_fields, magnetic_input_is_h);
    }
    else
    {
        writeNumpyOutput(output_dir, step, time, conv_plt, ic_dir,
                         grids, dmap, geom, efields, magnetic_fields,
                         magnetic_input_is_h);
    }
}
