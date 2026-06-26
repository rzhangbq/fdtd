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

// pec_location: 0 => wall PEC at pec_lo/pec_hi; otherwise interior PEC at that index.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool PecOnPlane(int pec_normal, int pec_location,
                                                       int pec_lo, int pec_hi, int i, int j,
                                                       int k) noexcept
{
    if (pec_location == 0)
    {
        return PecOnWallPlane(pec_normal, pec_lo, pec_hi, i, j, k);
    }

    if (pec_normal == 0)
    {
        return i == pec_location;
    }
    if (pec_normal == 1)
    {
        return j == pec_location;
    }
    return k == pec_location;
}

// Skip implicit / explicit E update and pin to zero on PEC planes.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool PecSkipEupdate(int pec_normal, int pec_location,
                                                             int e_comp,
                                                             bool nodal_on_pec_normal,
                                                             int pec_lo, int pec_hi, int i,
                                                             int j, int k) noexcept
{
    if (pec_normal < 0 || e_comp == pec_normal || !nodal_on_pec_normal)
    {
        return false;
    }
    return PecOnPlane(pec_normal, pec_location, pec_lo, pec_hi, i, j, k);
}

// Pin tangential nodal E to zero on PEC planes (walls or interior).
void PecPinTangentialE(int pec_normal, int pec_location,
                       amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &efields);

#endif
