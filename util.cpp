#include "util.H"

#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Utility.H>

#include <fstream>
#include <iomanip>
#include <sstream>

using namespace amrex;

namespace {
void writeVisitOutput (
    std::string const& output_dir,
    int step,
    Real time,
    BoxArray const& grids,
    DistributionMapping const& dmap,
    Geometry const& geom,
    Array<MultiFab,AMREX_SPACEDIM> const& efields,
    Array<MultiFab,AMREX_SPACEDIM> const& bfields)
{
    MultiFab plotmf(grids, dmap, 6, 0);
    UtilBuildCellCenteredPlotMF(plotmf, efields, bfields);

    const Vector<std::string> varnames = {"Ex", "Ey", "Ez", "Bx", "By", "Bz"};
    const std::string plotfile = Concatenate(output_dir, step);

    if (ParallelDescriptor::MyProc() == 0) {
        amrex::Print() << "Writing plotfile " << plotfile << " at time " << time << "\n";
    }

    WriteSingleLevelPlotfile(plotfile, plotmf, varnames, geom, time, step);
}

void writeNumpyOutput (
    std::string const& output_dir,
    int step,
    Real time,
    BoxArray const& grids,
    DistributionMapping const& dmap,
    Geometry const& geom,
    Array<MultiFab,AMREX_SPACEDIM> const& efields,
    Array<MultiFab,AMREX_SPACEDIM> const& bfields)
{
    MultiFab plotmf(grids, dmap, 6, 0);
    UtilBuildCellCenteredPlotMF(plotmf, efields, bfields);

    const Box& domain_box = geom.Domain();
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

    UtilCreateDirectory(output_dir, 0755);

    std::ostringstream step_tag;
    step_tag << std::setw(5) << std::setfill('0') << step;

    const std::string bin_file = output_dir + "/step_" + step_tag.str() + "_fields.bin";
    const std::string meta_file = output_dir + "/step_" + step_tag.str() + "_meta.json";

    {
        std::ofstream ofs(bin_file, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(data.data()),
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
    meta << "  \"components\": [\"Ex\", \"Ey\", \"Ez\", \"Bx\", \"By\", \"Bz\"],\n";
    meta << "  \"fields_file\": \"" << bin_file << "\",\n";
    meta << "  \"prob_lo\": [" << problo[0] << ", " << problo[1] << ", " << problo[2] << "],\n";
    meta << "  \"prob_hi\": [" << probhi[0] << ", " << probhi[1] << ", " << probhi[2] << "]\n";
    meta << "}\n";
}
} // namespace

void UtilBuildCellCenteredPlotMF (
    MultiFab& plotmf,
    Array<MultiFab,AMREX_SPACEDIM> const& efields,
    Array<MultiFab,AMREX_SPACEDIM> const& bfields)
{
    Vector<const MultiFab*> efield_ptrs{
        AMREX_D_DECL(&efields[0], &efields[1], &efields[2])
    };
    Vector<const MultiFab*> bfield_ptrs{
        AMREX_D_DECL(&bfields[0], &bfields[1], &bfields[2])
    };

    average_edge_to_cellcenter(plotmf, 0, efield_ptrs);
    average_face_to_cellcenter(plotmf, 3, bfield_ptrs);
}

void UtilWritePlotOutput (
    std::string const& plot_format,
    std::string const& output_dir,
    int step,
    Real time,
    BoxArray const& grids,
    DistributionMapping const& dmap,
    Geometry const& geom,
    Array<MultiFab,AMREX_SPACEDIM> const& efields,
    Array<MultiFab,AMREX_SPACEDIM> const& bfields)
{
    if (plot_format == "visit") {
        writeVisitOutput(output_dir, step, time, grids, dmap, geom, efields, bfields);
    } else {
        writeNumpyOutput(output_dir, step, time, grids, dmap, geom, efields, bfields);
    }
}
