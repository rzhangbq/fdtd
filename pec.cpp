#include "pec.H"

using namespace amrex;

void PecPinTangentialE(int pec_normal, int pec_location,
                       Array<MultiFab, AMREX_SPACEDIM> &efields)
{
    if (pec_normal < 0)
    {
        return;
    }

    for (int comp = 0; comp < AMREX_SPACEDIM; ++comp)
    {
        if (comp == pec_normal)
        {
            continue;
        }

        MultiFab &field = efields[comp];
        if (field.ixType().cellCentered(pec_normal))
        {
            continue;
        }

        Box const bounds = field.boxArray().minimalBox();
        int const lo = bounds.smallEnd(pec_normal);
        int const hi = bounds.bigEnd(pec_normal);

        auto const &arrs = field.arrays();

        ParallelFor(field, [=] AMREX_GPU_DEVICE(int b, int i, int j, int k) noexcept
                    {
            if (PecOnPlane(pec_normal, pec_location, lo, hi, i, j, k))
            {
                arrs[b](i, j, k) = 0.0_rt;
            } });
    }
}
