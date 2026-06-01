#ifndef HYDRO_RSOLVERS_LLF_HYD_SPLIT_HPP_
#define HYDRO_RSOLVERS_LLF_HYD_SPLIT_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_hyd_split.hpp
//! \brief Single-face LLF Riemann solver for the split-kernel flux path.
//! Reads L/R primitives from global per-face buffers and writes a single flux entry.
//! Templated on direction (ivx = IVX|IVY|IVZ).

#include "llf_hyd_singlestate.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
//! \fn SingleFaceLLF<ivx>()
//! \brief Compute the LLF flux at face (m,k,j,i) for direction ivx.
//!
//! Indexing convention for wl/wr (face-indexed in the face-normal axis,
//! cell-indexed in transverse axes, all origin-shifted by ks/js/is).
//!   ivx=IVX: face f stored at (k-ks, j-js, f-is) for f in [is, ie+1]
//!   ivx=IVY: face f stored at (k-ks, f-js, i-is) for f in [js, je+1]
//!   ivx=IVZ: face f stored at (f-ks, j-js, i-is) for f in [ks, ke+1]
//!
//! Flux is written to flx(m, n, k, j, i) using global (non-shifted) cell indices.
template <int ivx>
KOKKOS_INLINE_FUNCTION
void SingleFaceLLF(const EOS_Data &eos,
                   const int m, const int k, const int j, const int i,
                   const int is, const int js, const int ks,
                   const DvceArray5D<Real> &wl,
                   const DvceArray5D<Real> &wr,
                   const DvceArray5D<Real> &flx) {
  constexpr int ivy = IVX + ((ivx - IVX) + 1) % 3;
  constexpr int ivz = IVX + ((ivx - IVX) + 2) % 3;

  // Buffer-local face/cell indices
  const int kb = k - ks;
  const int jb = j - js;
  const int ib = i - is;

  // Extract L/R primitives at the face
  HydPrim1D wli, wri;
  wli.d  = wl(m, IDN, kb, jb, ib);
  wli.vx = wl(m, ivx, kb, jb, ib);
  wli.vy = wl(m, ivy, kb, jb, ib);
  wli.vz = wl(m, ivz, kb, jb, ib);

  wri.d  = wr(m, IDN, kb, jb, ib);
  wri.vx = wr(m, ivx, kb, jb, ib);
  wri.vy = wr(m, ivy, kb, jb, ib);
  wri.vz = wr(m, ivz, kb, jb, ib);

  if (eos.is_ideal) {
    wli.e = wl(m, IEN, kb, jb, ib);
    wri.e = wr(m, IEN, kb, jb, ib);
  }

  // Call single-state LLF (math identical to the existing wrapper)
  HydCons1D flux;
  SingleStateLLF_Hyd(wli, wri, eos, flux);

  // Store
  flx(m, IDN, k, j, i) = flux.d;
  flx(m, ivx, k, j, i) = flux.mx;
  flx(m, ivy, k, j, i) = flux.my;
  flx(m, ivz, k, j, i) = flux.mz;
  if (eos.is_ideal) { flx(m, IEN, k, j, i) = flux.e; }
}

} // namespace hydro
#endif  // HYDRO_RSOLVERS_LLF_HYD_SPLIT_HPP_
