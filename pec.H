#ifndef PEC_H_
#define PEC_H_

#include <AMReX_Array.H>
#include <AMReX_Gpu.H>
#include <AMReX_MultiFab.H>

// True when (i,j,k) lies on a PEC wall plane for tangential E that is nodal along pec_normal.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool PecOnWallPlane(int pec_normal, int pec_lo,
                                                               int pec_hi, int i, int j,
                                                               int k) noexcept
{
    if (pec_normal == 0)
    {
        return i == pec_lo || i == pec_hi;
    }
    if (pec_normal == 1)
    {
        return j == pec_lo || j == pec_hi;
    }
    return k == pec_lo || k == pec_hi;
}

// Skip implicit / explicit E update and pin to zero on PEC wall planes.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool PecSkipEupdate(int pec_normal, int e_comp,
                                                             bool nodal_on_pec_normal,
                                                             int pec_lo, int pec_hi, int i,
                                                             int j, int k) noexcept
{
    if (pec_normal < 0 || e_comp == pec_normal || !nodal_on_pec_normal)
    {
        return false;
    }
    return PecOnWallPlane(pec_normal, pec_lo, pec_hi, i, j, k);
}

// Pin tangential nodal E to zero on PEC walls (used once after initial data).
void PecPinTangentialEwalls(int pec_normal,
                            amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &efields);

#endif
