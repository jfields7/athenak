#ifndef RECONSTRUCT_RECON_HPP_
#define RECONSTRUCT_RECON_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file recon.hpp
//! \brief Per-cell reconstruction supporting every reconstruction method (DC, PLM, PPM4,
//! PPMX, WENOZ).  Each cell produces ql at its right face and qr at its left face,
//! written to the global per-face L/R buffers.  Templated on direction
//! (ivx = IVX|IVY|IVZ); the method is selected at runtime via a grid-uniform branch
//! (coherent across threads, so effectively branch-free on GPU).

#include "athena.hpp"
#include "eos/eos.hpp"
#include "dc.hpp"     // (trivial donor-cell handled inline below)
#include "plm.hpp"    // PLM()
#include "ppm.hpp"    // PPM4(), PPMX()
#include "wenoz.hpp"  // WENOZ()

//----------------------------------------------------------------------------------------
//! \fn ReconCell<ivx>()
//! \brief Single-cell reconstruction for the requested method.  Reads the stencil
//! centered on cell (m,k,j,i) in direction `ivx`, and writes ql to face (i+1) and qr to
//! face (i) in the global per-face buffers.  Buffers are indexed by the GLOBAL cell/face
//! index (m,n,k,j,i); they are sized to the full cell range (including ghosts) so any
//! face touched by the reconstruction loop is in bounds (requires nghost >= 2).  Caller
//! loops over the cells whose faces are needed (one extra cell on the normal axis):
//!   ivx=IVX:  i in [is-1, ie+1], (transverse j,k over their active+ghost range)
//!   ivx=IVY:  j in [js-1, je+1], (transverse i,k over their active+ghost range)
//!   ivx=IVZ:  k in [ks-1, ke+1], (transverse i,j over their active+ghost range)
//!
//! The wide (+/-2) stencil used by PPM/WENOZ requires nghost >= 3 (enforced in the
//! Hydro/MHD constructor); DC/PLM only touch the +/-1 stencil.
template <int ivx>
KOKKOS_INLINE_FUNCTION
void ReconCell(const ReconstructionMethod recon, const EOS_Data &eos,
               const bool apply_floors,
               const int m, const int k, const int j, const int i,
               const int nvars,
               const DvceArray5D<Real> &q,
               const DvceArray5D<Real> &wl,
               const DvceArray5D<Real> &wr) {
  // Compile-time stencil offsets along the reconstruction direction
  constexpr int di = (ivx == IVX) ? 1 : 0;
  constexpr int dj = (ivx == IVY) ? 1 : 0;
  constexpr int dk = (ivx == IVZ) ? 1 : 0;

  const Real dfloor = eos.dfloor;
  const Real efloor = eos.is_ideal ? (eos.pfloor/(eos.gamma - 1.0)) : 0.0;

  for (int n = 0; n < nvars; ++n) {
    Real ql_val, qr_val;
    switch (recon) {
      case ReconstructionMethod::dc:
        ql_val = q(m, n, k, j, i);
        qr_val = q(m, n, k, j, i);
        break;
      case ReconstructionMethod::plm:
        PLM(q(m, n, k - dk, j - dj, i - di),
            q(m, n, k,      j,      i),
            q(m, n, k + dk, j + dj, i + di),
            ql_val, qr_val);
        break;
      case ReconstructionMethod::ppm4:
        PPM4(q(m, n, k - 2*dk, j - 2*dj, i - 2*di),
             q(m, n, k -   dk, j -   dj, i -   di),
             q(m, n, k,        j,        i),
             q(m, n, k +   dk, j +   dj, i +   di),
             q(m, n, k + 2*dk, j + 2*dj, i + 2*di),
             ql_val, qr_val);
        break;
      case ReconstructionMethod::ppmx:
        PPMX(q(m, n, k - 2*dk, j - 2*dj, i - 2*di),
             q(m, n, k -   dk, j -   dj, i -   di),
             q(m, n, k,        j,        i),
             q(m, n, k +   dk, j +   dj, i +   di),
             q(m, n, k + 2*dk, j + 2*dj, i + 2*di),
             ql_val, qr_val);
        if (apply_floors) {
          if (n == IDN) { ql_val = fmax(ql_val, dfloor); qr_val = fmax(qr_val, dfloor); }
          if (eos.is_ideal && n == IEN) {
            ql_val = fmax(ql_val, efloor); qr_val = fmax(qr_val, efloor);
          }
        }
        break;
      case ReconstructionMethod::wenoz:
        WENOZ(q(m, n, k - 2*dk, j - 2*dj, i - 2*di),
              q(m, n, k -   dk, j -   dj, i -   di),
              q(m, n, k,        j,        i),
              q(m, n, k +   dk, j +   dj, i +   di),
              q(m, n, k + 2*dk, j + 2*dj, i + 2*di),
              ql_val, qr_val);
        if (apply_floors) {
          if (n == IDN) { ql_val = fmax(ql_val, dfloor); qr_val = fmax(qr_val, dfloor); }
          if (eos.is_ideal && n == IEN) {
            ql_val = fmax(ql_val, efloor); qr_val = fmax(qr_val, efloor);
          }
        }
        break;
      default:
        ql_val = q(m, n, k, j, i);
        qr_val = q(m, n, k, j, i);
        break;
    }
    // ql is the right-face value of cell (writes the LEFT state of face i+1);
    // qr is the left-face value of cell (writes the RIGHT state of face i).
    wl(m, n, k + dk, j + dj, i + di) = ql_val;
    wr(m, n, k, j, i) = qr_val;
  }
}

#endif  // RECONSTRUCT_RECON_HPP_
