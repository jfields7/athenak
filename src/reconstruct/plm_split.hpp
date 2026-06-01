#ifndef RECONSTRUCT_PLM_SPLIT_HPP_
#define RECONSTRUCT_PLM_SPLIT_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file plm_split.hpp
//! \brief Per-cell PLM reconstruction for the split-kernel flux path.  Each cell
//! produces ql at its right face and qr at its left face, written to global
//! per-face L/R buffers.  Templated on direction (ivx = IVX|IVY|IVZ).

#include "athena.hpp"
#include "plm.hpp"  // for inline PLM() scalar kernel

//----------------------------------------------------------------------------------------
//! \fn PiecewiseLinearSplit<ivx>()
//! \brief Single-cell PLM reconstruction.  Reads q at the 3-point stencil centered on
//! cell (m,k,j,i) in direction `ivx`, and writes ql to face (i+1) and qr to face (i)
//! in the global per-face buffers.  Buffer indexing convention (face-indexed in the
//! face-normal axis, cell-indexed in transverse axes, all origin-shifted by
//! ks/js/is so the buffer's local coordinate is in [0, nx+...]).
//!
//! Out-of-bounds face writes at the extreme cells are predicate-skipped.  Active
//! face range:  ivx=IVX -> [is, ie+1];  ivx=IVY -> [js, je+1];  ivx=IVZ -> [ks, ke+1].
//!
//! Caller is expected to loop cells:
//!   ivx=IVX:  i in [is-1, ie+1], j in [js, je], k in [ks, ke]
//!   ivx=IVY:  i in [is, ie],     j in [js-1, je+1], k in [ks, ke]
//!   ivx=IVZ:  i in [is, ie],     j in [js, je], k in [ks-1, ke+1]
template <int ivx>
KOKKOS_INLINE_FUNCTION
void PiecewiseLinearSplit(const int m, const int k, const int j, const int i,
                          const int is, const int js, const int ks,
                          const int ie, const int je, const int ke,
                          const int nvars,
                          const DvceArray5D<Real> &q,
                          const DvceArray5D<Real> &wl,
                          const DvceArray5D<Real> &wr) {
  // Compile-time stencil offsets and face-axis bounds
  constexpr int di = (ivx == IVX) ? 1 : 0;
  constexpr int dj = (ivx == IVY) ? 1 : 0;
  constexpr int dk = (ivx == IVZ) ? 1 : 0;

  // Map (k,j,i) global cell index -> buffer local index (origin-shifted by ks/js/is)
  const int kb = k - ks;
  const int jb = j - js;
  const int ib = i - is;

  // Face-axis index for the OUTGOING ql (face on the +ivx side of cell): cell coord+1
  // Face-axis index for the OUTGOING qr (face on the -ivx side of cell): cell coord
  // In buffer-local coords, these are (kb+dk, jb+dj, ib+di) and (kb, jb, ib) respectively.
  // Predicate: a face slot is in-bounds iff its face-axis index is in [0, nx_dir].
  const int face_dim = (ivx == IVX) ? (ie - is + 1)
                     : (ivx == IVY) ? (je - js + 1)
                                    : (ke - ks + 1);  // nx_dir
  const int ql_face = (ivx == IVX) ? (ib + 1)
                    : (ivx == IVY) ? (jb + 1)
                                   : (kb + 1);
  const int qr_face = (ivx == IVX) ? ib
                    : (ivx == IVY) ? jb
                                   : kb;

  const bool ql_ok = (ql_face >= 0) && (ql_face <= face_dim);
  const bool qr_ok = (qr_face >= 0) && (qr_face <= face_dim);

  for (int n = 0; n < nvars; ++n) {
    Real ql_val, qr_val;
    PLM(q(m, n, k - dk, j - dj, i - di),
        q(m, n, k,      j,      i),
        q(m, n, k + dk, j + dj, i + di),
        ql_val, qr_val);
    if (ql_ok) {
      wl(m, n, kb + dk, jb + dj, ib + di) = ql_val;
    }
    if (qr_ok) {
      wr(m, n, kb, jb, ib) = qr_val;
    }
  }
}

#endif  // RECONSTRUCT_PLM_SPLIT_HPP_
