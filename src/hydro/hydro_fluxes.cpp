//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_fluxes.cpp
//! \brief Calculate 3D fluxes for hydro.
//!
//! Fluxes are computed with two 1D-RangePolicy kernels per direction: (1) a per-cell
//! reconstruction kernel that materializes the L/R primitive states in the global
//! wl_split/wr_split buffers, followed by (2) a per-face Riemann solve that reads those
//! buffers and writes the interface flux.  All reconstruction methods (DC/PLM/PPM4/
//! PPMX/WENOZ) and non-relativistic Riemann solvers (Advect/LLF/HLLE/HLLC/Roe) are
//! supported; the reconstruction method is chosen at runtime, the solver at compile time
//! via the rsolver template parameter.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "hydro.hpp"
#include "eos/eos.hpp"
#include "reconstruct/recon.hpp"
#include "hydro/rsolvers/advect_hyd.hpp"
#include "hydro/rsolvers/llf_hyd.hpp"
#include "hydro/rsolvers/hlle_hyd.hpp"
#include "hydro/rsolvers/hllc_hyd.hpp"
#include "hydro/rsolvers/roe_hyd.hpp"
#include "hydro/rsolvers/llf_srhyd.hpp"
#include "hydro/rsolvers/hlle_srhyd.hpp"
#include "hydro/rsolvers/hllc_srhyd.hpp"
#include "hydro/rsolvers/llf_grhyd.hpp"
#include "hydro/rsolvers/hlle_grhyd.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
//! \fn SolveFace<ivx>()
//! \brief Dispatch the (compile-time) Riemann solver for a single face.  Capturing the
//! solver inputs into locals before the constexpr-if is required for CUDA 11.6+.
template <Hydro_RSolver rsolver_method_, int ivx>
KOKKOS_INLINE_FUNCTION
void SolveFace(const EOS_Data &eos, const RegionIndcs &indcs,
               const DualArray1D<RegionSize> &size, const CoordData &coord,
               const int m, const int k, const int j, const int i,
               const int is, const int js, const int ks,
               const DvceArray5D<Real> &wl,
               const DvceArray5D<Real> &wr,
               const DvceArray5D<Real> &flx) {
  if constexpr (rsolver_method_ == Hydro_RSolver::advect) {
    Advect<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, flx);
  } else if constexpr (rsolver_method_ == Hydro_RSolver::llf) {
    LLF<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, flx);
  } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle) {
    HLLE<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, flx);
  } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc) {
    HLLC<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, flx);
  } else if constexpr (rsolver_method_ == Hydro_RSolver::roe) {
    Roe<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, flx);
  } else if constexpr (rsolver_method_ == Hydro_RSolver::llf_sr) {
    LLF_SR<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, flx);
  } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_sr) {
    HLLE_SR<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, flx);
  } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc_sr) {
    HLLC_SR<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, flx);
  } else if constexpr (rsolver_method_ == Hydro_RSolver::llf_gr) {
    LLF_GR<ivx>(eos, indcs, size, coord, m, k, j, i, is, js, ks, wl, wr, flx);
  } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_gr) {
    HLLE_GR<ivx>(eos, indcs, size, coord, m, k, j, i, is, js, ks, wl, wr, flx);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Hydro::CalculateFluxes
//! \brief Calls reconstruction and Riemann solver functions to compute hydro fluxes.
//! Templated over the Riemann solver for better performance on GPUs.

template <Hydro_RSolver rsolver_method_>
void Hydro::CalculateFluxes(Driver *pdriver, int stage) {
  RegionIndcs &indcs_ = pmy_pack->pmesh->mb_indcs;
  int is = indcs_.is, ie = indcs_.ie;
  int js = indcs_.js, je = indcs_.je;
  int ks = indcs_.ks, ke = indcs_.ke;

  int &nhyd_  = nhydro;
  int nvars = nhydro + nscalars;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  const auto recon_method_ = recon_method;

  auto &eos_ = peos->eos_data;
  auto &size_ = pmy_pack->pmb->mb_size;
  auto &coord_ = pmy_pack->pcoord->coord_data;
  auto &w0_ = w0;
  auto wl_ = wl_split;
  auto wr_ = wr_split;

  //------------------------------------------------------------------------------------
  // x1 direction
  {
    auto &flx1 = uflx.x1f;
    // Reconstruction over cells [is-1, ie+1], all j in [js, je], all k in [ks, ke]
    par_for("hflux_x1_recon", DevExeSpace(),
      0, nmb1, ks, ke, js, je, is-1, ie+1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        ReconCell<IVX>(recon_method_, eos_, true, m, k, j, i, is, js, ks, ie, je, ke,
                       nvars, w0_, wl_, wr_);
      });

    // Riemann solve over faces [is, ie+1]
    par_for("hflux_x1_rsolve", DevExeSpace(),
      0, nmb1, ks, ke, js, je, is, ie+1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        auto eos = eos_;
        auto indcs = indcs_;
        auto size = size_;
        auto coord = coord_;
        auto wl = wl_;
        auto wr = wr_;
        auto flx = flx1;
        const int is_ = is, js_ = js, ks_ = ks;
        SolveFace<rsolver_method_, IVX>(eos, indcs, size, coord,
                                        m, k, j, i, is_, js_, ks_, wl, wr, flx);
      });

    // Scalar fluxes (upwind from sign of mass flux)
    if (nvars > nhyd_) {
      par_for("hflux_x1_scalars", DevExeSpace(),
        0, nmb1, ks, ke, js, je, is, ie+1,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          for (int n = nhyd_; n < nvars; ++n) {
            if (flx1(m, IDN, k, j, i) >= 0.0) {
              flx1(m, n, k, j, i) = flx1(m, IDN, k, j, i) * wl_(m, n, k, j, i);
            } else {
              flx1(m, n, k, j, i) = flx1(m, IDN, k, j, i) * wr_(m, n, k, j, i);
            }
          }
        });
    }
  }

  //------------------------------------------------------------------------------------
  // x2 direction
  if (pmy_pack->pmesh->multi_d) {
    auto &flx2 = uflx.x2f;
    // Reconstruction over cells j in [js-1, je+1], all i in [is, ie], all k in [ks, ke]
    par_for("hflux_x2_recon", DevExeSpace(),
      0, nmb1, ks, ke, js-1, je+1, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        ReconCell<IVY>(recon_method_, eos_, true, m, k, j, i, is, js, ks, ie, je, ke,
                       nvars, w0_, wl_, wr_);
      });

    // Riemann solve over faces j in [js, je+1]
    par_for("hflux_x2_rsolve", DevExeSpace(),
      0, nmb1, ks, ke, js, je+1, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        auto eos = eos_;
        auto indcs = indcs_;
        auto size = size_;
        auto coord = coord_;
        auto wl = wl_;
        auto wr = wr_;
        auto flx = flx2;
        const int is_ = is, js_ = js, ks_ = ks;
        SolveFace<rsolver_method_, IVY>(eos, indcs, size, coord,
                                        m, k, j, i, is_, js_, ks_, wl, wr, flx);
      });

    if (nvars > nhyd_) {
      par_for("hflux_x2_scalars", DevExeSpace(),
        0, nmb1, ks, ke, js, je+1, is, ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          for (int n = nhyd_; n < nvars; ++n) {
            if (flx2(m, IDN, k, j, i) >= 0.0) {
              flx2(m, n, k, j, i) = flx2(m, IDN, k, j, i) * wl_(m, n, k, j, i);
            } else {
              flx2(m, n, k, j, i) = flx2(m, IDN, k, j, i) * wr_(m, n, k, j, i);
            }
          }
        });
    }
  }

  //------------------------------------------------------------------------------------
  // x3 direction
  if (pmy_pack->pmesh->three_d) {
    auto &flx3 = uflx.x3f;
    // Reconstruction over cells k in [ks-1, ke+1], all j in [js, je], all i in [is, ie]
    par_for("hflux_x3_recon", DevExeSpace(),
      0, nmb1, ks-1, ke+1, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        ReconCell<IVZ>(recon_method_, eos_, true, m, k, j, i, is, js, ks, ie, je, ke,
                       nvars, w0_, wl_, wr_);
      });

    // Riemann solve over faces k in [ks, ke+1]
    par_for("hflux_x3_rsolve", DevExeSpace(),
      0, nmb1, ks, ke+1, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        auto eos = eos_;
        auto indcs = indcs_;
        auto size = size_;
        auto coord = coord_;
        auto wl = wl_;
        auto wr = wr_;
        auto flx = flx3;
        const int is_ = is, js_ = js, ks_ = ks;
        SolveFace<rsolver_method_, IVZ>(eos, indcs, size, coord,
                                        m, k, j, i, is_, js_, ks_, wl, wr, flx);
      });

    if (nvars > nhyd_) {
      par_for("hflux_x3_scalars", DevExeSpace(),
        0, nmb1, ks, ke+1, js, je, is, ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          for (int n = nhyd_; n < nvars; ++n) {
            if (flx3(m, IDN, k, j, i) >= 0.0) {
              flx3(m, n, k, j, i) = flx3(m, IDN, k, j, i) * wl_(m, n, k, j, i);
            } else {
              flx3(m, n, k, j, i) = flx3(m, IDN, k, j, i) * wr_(m, n, k, j, i);
            }
          }
        });
    }
  }

  return;
}

// function definitions for each template parameter
template void Hydro::CalculateFluxes<Hydro_RSolver::advect>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::llf>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::hlle>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::hllc>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::roe>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::llf_sr>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::hlle_sr>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::hllc_sr>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::llf_gr>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::hlle_gr>(Driver *pdriver, int stage);

} // namespace hydro
