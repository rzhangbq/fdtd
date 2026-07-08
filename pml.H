#ifndef PML_H_
#define PML_H_

#include <AMReX_BoxArray.H>
#include <AMReX_MultiFab.H>

// Build a coefficient MultiFab on the destination field/psi layout. The source
// PML profile is cell-centered, so nodal boundary values may not be covered by
// ParallelCopy on non-periodic boundaries; initialize those to the analytic
// non-PML defaults before copying.
inline amrex::MultiFab
PmlMakeCoeffLike(amrex::MultiFab const &field, amrex::Real initial_value)
{
    amrex::MultiFab mf(field.boxArray(), field.DistributionMap(), 1, field.nGrowVect());
    mf.setVal(initial_value);
    return mf;
}

// Copy coefficient data into a target layout (same helper pattern as adi.cpp/pec split).
inline void
PmlCopyCoefToLayout(amrex::MultiFab &dst, amrex::MultiFab const &src,
                    amrex::Periodicity const &period)
{
    dst.ParallelCopy(src, 0, 0, 1, amrex::IntVect(0), dst.nGrowVect(), period);
}

#endif
