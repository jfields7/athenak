#ifndef DYN_GRMHD_DYN_GRMHD_WB_HPP_
#define DYN_GRMHD_DYN_GRMHD_WB_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyn_grmhd_wb.hpp
//! \brief Inline functions for computing hydrostatic equilibrium for well balancing.

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
#include "eos/primitive_solver_hyd.hpp"
#include "reconstruct/recon.hpp"

// A symmetric interpolation operator to compute the interface i+1/2.
template<int ivx, int nghosts, int sign, class Arr, class... Idxs>
KOKKOS_INLINE_FUNCTION
decltype(auto) InterpToInterface(const Arr& q,
                                 const int k, const int j, const int i, Idxs... idxs) {
  constexpr int shifti = sign*(ivx == IVX);
  constexpr int shiftj = sign*(ivx == IVY);
  constexpr int shiftk = sign*(ivx == IVZ);
  if constexpr (nghosts == 2) {
    return Real(0.5)*(q(idxs..., k, j, i) + q(idxs..., k+shiftk, j+shiftj, i+shifti));
  } else if constexpr (nghosts == 3) {
    return -Real(1./16.)*(q(idxs..., k-shiftk, j-shiftj, i-shifti) +
                          q(idxs..., k+2*shiftk, j+2*shiftj, i+2*shifti))
           +Real(9./16.)*(q(idxs..., k, j, i) +
                          q(idxs..., k+shiftk, j+shiftj, i+shifti));
  } else if constexpr (nghosts == 4) {
    return +Real(3./256.  )*(q(idxs..., k-2*shiftk, j-2*shiftj, i-2*shifti) +
                             q(idxs..., k+3*shiftk, j+3*shiftj, i+3*shifti))
           -Real(25./256. )*(q(idxs..., k-shiftk, j-shiftj, i-shifti) +
                             q(idxs..., k+2*shiftk, j+2*shiftj, i+2*shifti))
           +Real(150./256.)*(q(idxs..., k, j, i) +
                             q(idxs..., k+shiftk, j+shiftj, i+shifti));
  } else {
    static_assert(!sizeof(Arr*), "Unsupported nghosts requested for InterpToInterface.");
  }
}

// An interpolation operator to compute the interface i+3/2. This is for working with the
// local approximation
template<int ivx, int nghosts, int sign, class Arr, class... Idxs>
decltype(auto) InterpToNextInterface(const Arr& q,
                                 const int k, const int j, const int i, Idxs... idxs) {
  constexpr int shifti = sign*(ivx == IVX);
  constexpr int shiftj = sign*(ivx == IVY);
  constexpr int shiftk = sign*(ivx == IVZ);
  if constexpr (nghosts == 2) {
    // In the 2nd-order case, we will not exceed the reconstruction stencil, so there's no
    // issue with using both neighbors
    return Real(0.5)*(q(idxs..., k+shiftk, j+shiftj, i+shifti) +
                      q(idxs..., k+2*shiftk, j+2*shiftj, i+2*shifti));
  } else if constexpr (nghosts == 3) {
    // In the 4th-order case, we interpolate a cubic polynomial centered on i+1/2.
    return (Real(1./16.)*q(idxs..., k-shiftk, j-shiftj, i-shifti) -
            Real(5./16.)*q(idxs..., k, j, i)) +
           (Real(15./16.)*q(idxs..., k+shiftk, j+shiftj, i+shifti) +
            Real(5./16.)*q(idxs..., k+2*shiftk, j+2*shiftj, i+2*shifti));
  } else if constexpr (nghosts == 4) {
    // In the 6th-order case, we interpolate a quintic polynomial centered on i+1/2.
    return (Real(7./256.)*q(idxs..., k-2*shiftk, j-2*shiftj, i-2*shifti) -
            Real(45./256.)*q(idxs..., k-shiftk, j-shiftj, i-shifti)) +
           (Real(126./256.)*q(idxs..., k, j, i) +
            Real(63./256.)*q(idxs..., k+3*shiftk, j+3*shiftj, i+3*shifti)) +
           (-Real(210./256.)*q(idxs..., k+shiftk, j+shiftj, i+shifti) +
             Real(315./256.)*q(idxs..., k+2*shiftk, j+2*shiftj, i+2*shifti));
  } else {
    static_assert(!sizeof(Arr*),
                  "Unsupported nghosts required for InterpToNextInterface.");
  }
}

//! \struct WBStateCenter
//! \brief Store a cell-centered state needed for the well-balanced scheme.
struct WBStateCenter {
  // Metric variables
  Real alp;
  Real sdetg;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> gdd;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> guu;

  // Fluid variables
  Real Phat;
  Real etild;

  template<class EOSPolicy, class ErrorPolicy>
  KOKKOS_INLINE_FUNCTION
  WBStateCenter(const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& pseos,
                const adm::ADM::ADM_vars& adm,
                const DvceArray5D<Real> temp, const DvceArray5D<Real> w0,
                const int nscal, const int m, const int k, const int j, const int i) {
    SetState(pseos, adm, temp, w0, nscal, m, k, j, i);
  }

  template<class EOSPolicy, class ErrorPolicy>
  KOKKOS_INLINE_FUNCTION
  void SetState(const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& pseos,
                const adm::ADM::ADM_vars& adm,
                const DvceArray5D<Real> temp, const DvceArray5D<Real> w0,
                const int nscal, const int m, const int k, const int j, const int i) {
    // Extract or compute metric quantities
    alp = adm.alpha(m, k, j, i);
    for (int b = 0; b < 3; b++) {
      for (int a = b; a < 3; a++) {
        gdd(b, a) = adm.g_dd(m, b, a, k, j, i);
      }
    }
    sdetg = Kokkos::sqrt(adm::SpatialDet(gdd(0, 0), gdd(0, 1), gdd(0, 2),
                                         gdd(1, 1), gdd(1, 2), gdd(2, 2)));
    adm::SpatialInv(1.0/(sdetg*sdetg), gdd(0, 0), gdd(0, 1), gdd(0, 2),
                    gdd(1, 1), gdd(1, 2), gdd(2, 2),
                    &guu(0, 0), &guu(0, 1), &guu(0, 2),
                    &guu(1, 1), &guu(1, 2), &guu(2, 2));

    // Densitize the pressure and compute the densitized energy density
    Real n = w0(m, IDN, k, j, i)/pseos.ps.GetEOS().GetBaryonMass();
    Phat = alp*sdetg*w0(m, IPR, k, j, i);
    Real Y[MAX_SPECIES];
    for (int s = 0; s < nscal; s++) {
      Y[s] = w0(m, IYF + s, k, j, i);
    }
    etild = sdetg*pseos.ps.GetEOS().GetEnergy(n, temp(m, 0, k, j, i), Y);
  }
};

//! \fn IntegrateEquilibrium
//! \brief Compute the equilibrium by integrating over the interval [x_0, x_1]
//
// The computation approximates the following integral:
// \hat{P}_{eq}(x_1) = \hat{P}(x_0) +
//         \frac{1}{2}\int_{x_0}^{x_1} \hat{P}\gamma^{a b} \partial_x \gamma_{a b} dx -
//         \int_{x_0}^{x_1}\tilde{e} \partial_x \alpha dx,
// where hats mark variables densitized by \alpha\sqrt{\gamma} and tildes indicate
// variables densitized by \sqrt{\gamma}. The approximation is as follows:
//
// \hat{P}_{eq} = \hat{P}_0 +
//     \frac{1}{2} \hat{P}_1 \gamma_1^{a b} (\gamma^1_{a b} - \gamma^0_{a b}) -
//     \tilde{e}(\alpha_1 - \alpha_0)
//
// This corresponds to assuming that \hat{P}\gamma^{a b} and \tilde{e} are constant over
// the interval [x_0, x_1] and integrating \partial_x \gamma_{a b} and \partial_x \alpha
// exactly. In isolation this is only first-order accurate, but in practice we have
// x_0 = x_i or x_{i\pm1/2} and x_1 = x_{i\pm1/2} or x_{i\pm1}, and we assume that
// \hat{P}\gamma^{a b} and \tilde{e} are constant over the interval [x_i, x_{i\pm1}], so
// this ends up being second-order accurate when taken as a whole.
//
// Arguments:
//   Peqhat0: the spacetime-densitized equilibrium pressure at x_0
//   Phat1: the spacetime-densitized pressure at x_1
//   etild1: the volume-densitized total energy density at x_1
//   alp0: the lapse at x_0
//   alp1: the lapse at x_1
//   gdd0: \gamma_{a b} at x_0
//   gdd1: \gamma_{a b} at x_1
//   guu1: \gamma^{a b} at x_1
//
// Returns: the equilibrium pressure integrated to x_1.
//
KOKKOS_INLINE_FUNCTION
Real IntegrateEquilibrium(const Real Peqhat0, const Real Phat1, const Real etild1,
                          const Real alp0, const Real alp1,
                          const AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2>& gdd0,
                          const AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2>& gdd1,
                          const AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2>& guu1) {
  Real tracediff = 0.0;
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < 3; b++) {
      tracediff += guu1(a, b)*(gdd1(a, b) - gdd0(a, b));
    }
  }
  return Peqhat0 + 0.5*Phat1*tracediff - etild1*(alp1 - alp0);
}

// This is needed because the constructor to WBStateInterface is templated; while
// constructors can be templated, their template arguments must be inferred rather than
// passed explicitly. This policy is a dummy object passed so that the arguments can be
// inferred.
template<int ivx, int nghosts, int shift>
struct InterfacePolicy {};

struct WBStateInterface {
  Real alp;
  Real Peqhat;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> gdd;

  template<int ivx, int nghosts, int shift>
  KOKKOS_INLINE_FUNCTION
  WBStateInterface(InterfacePolicy<ivx, nghosts, shift> policy,
                   const adm::ADM::ADM_vars& adm, const WBStateCenter& state,
                   const Real& Phateq, const int m, const int k, const int j,
                   const int i) {
    constexpr int sign = (shift > 0) ? 1 : -1;
    if constexpr (shift == 1 || shift == -1) {
      alp = InterpToInterface<ivx, nghosts, sign>(adm.alpha, k, j, i, m);
      for (int b = 0; b < 3; b++) {
        for (int a = b; a < 3; a++) {
          gdd(b, a) = InterpToInterface<ivx, nghosts, sign>(adm.g_dd, k, j, i, m, b, a);
        }
      }
    } else {
      // This convoluted logic is for when we need to compute interfaces at i=+/- 3/2.
      // It could potentially also be used for higher-order calculations, too.
      constexpr int di = sign*(ivx == IVX)*(shift*sign - 1);
      constexpr int dj = sign*(ivx == IVY)*(shift*sign - 1);
      constexpr int dk = sign*(ivx == IVZ)*(shift*sign - 1);
      alp = InterpToInterface<ivx, nghosts, sign>(adm.alpha, k+dk, j+dj, i+di, m);
      for (int b = 0; b < 3; b++) {
        for (int a = b; a < 3; a++) {
          gdd(b, a) = InterpToInterface<ivx, nghosts, sign>(adm.g_dd,
                          k+dk, j+dj, i+di, m, b, a);
        }
      }
    }
    Peqhat = IntegrateEquilibrium(Phateq, state.Phat, state.etild,
                                  state.alp, alp, state.gdd, gdd, state.guu);
  }

  KOKKOS_INLINE_FUNCTION
  Real ComputeSpacetimeVolume() const {
    return alp*Kokkos::sqrt(adm::SpatialDet(gdd(0,0), gdd(0,1), gdd(0,2),
                                            gdd(1,1), gdd(1,2), gdd(2,2)));
  }
};

template<ReconstructionMethod recon, int ivx, class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void WellBalancedCellT(const int m, const int k, const int j, const int i,
                       const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& pseos,
                       const DvceArray5D<Real> temp, const adm::ADM::ADM_vars& adm,
                       const DvceArray5D<Real> w0,
                       const DvceArray5D<Real> wl, const DvceArray5D<Real> wr,
                       const int nscal) {
  WBStateCenter csi{pseos, adm, temp, w0, nscal, m, k, j, i};

  // Regardless of the method, we need to compute the equilibrium pressure at the left and
  // right interfaces for this cell

  // LEFT INTERFACE
  WBStateInterface ip12{InterfacePolicy<ivx, 2, 1>(), adm, csi, csi.Phat, m, k, j, i};
  if (ip12.Peqhat < 0.0) {
    // Abort the equilibrium calculation if there is no equilibrium.
    return;
  }
  // We'll need this later on to undensitize stuff
  Real volp12 = ip12.ComputeSpacetimeVolume();
  if (volp12 < Kokkos::Experimental::epsilon_v<Real>) {
    // Abort the equilibrium calculation if the spacetime volume is too small; we're
    // at a singularity.
    return;
  }

  // RIGHT INTERFACE
  WBStateInterface im12{InterfacePolicy<ivx, 2, -1>(), adm, csi, csi.Phat, m, k, j, i};
  if (im12.Peqhat < 0.0) {
    // Abort the equilibrium calculation if there is no equilibrium.
    return;
  }
  // We'll need this later on to undensitize stuff
  Real volm12 = im12.ComputeSpacetimeVolume();
  if (volm12 < Kokkos::Experimental::epsilon_v<Real>) {
    // Abort the equilibrium calculation if the spacetime volume is too small; we're
    // at a singularity.
    return;
  }

  constexpr int di = (ivx == IVX);
  constexpr int dj = (ivx == IVY);
  constexpr int dk = (ivx == IVZ);
  if constexpr (recon == ReconstructionMethod::dc) {
    // Apply donor-cell reconstruction here.
    wl(m, IPR, k + dk, j + dj, i + di) = ip12.Peqhat/volp12;
    wr(m, IPR, k, j, i) = im12.Peqhat/volm12;
    return;
  } else {
    // For second-order reconstruction, we need to integrate to the neighboring cells.

    // Integrate to i+1
    csi.SetState(pseos, adm, temp, w0, nscal, m, k+dk, j+dj, i+di);
    Real Peqhatp1 = IntegrateEquilibrium(ip12.Peqhat, csi.Phat, csi.etild, ip12.alp,
                                         csi.alp, ip12.gdd, csi.gdd, csi.guu);
    if (Peqhatp1 < 0.0) {
      // Abort the equilibrium calculation if there is no equilibrium.
      return;
    }
    // Store these values now so that the compiler can clear much larger objects from
    // register space.
    Real Peqhatp12 = ip12.Peqhat;
    Real dPhatp1 = csi.Phat - Peqhatp1;

    Real dPhatp2;
    // Only continue integrating if we're not using PLM.
    if constexpr (recon != ReconstructionMethod::plm) {
      // Integrate to the i+3/2 interface
      WBStateInterface ip32{InterfacePolicy<ivx,2,2>(), adm, csi, Peqhatp1, m, k, j, i};
      if (ip32.Peqhat < 0.0) {
        return;
      }

      // Integrate to the i+2 cell
      csi.SetState(pseos, adm, temp, w0, nscal, m, k+2*dk, j+2*dj, i+2*di);
      Real Peqhatp2 = IntegrateEquilibrium(ip32.Peqhat, csi.Phat, csi.etild, ip32.alp,
                                           csi.alp, ip32.gdd, csi.gdd, csi.guu);
      if (Peqhatp2 < 0.0) {
        // Abort the equilibrium calculation if there is no equilibrium.
        return;
      }
      dPhatp2 = csi.Phat - Peqhatp2;
    }

    // Integrate to i-1
    csi.SetState(pseos, adm, temp, w0, nscal, m, k-dk, j-dj, i-di);
    Real Peqhatm1 = IntegrateEquilibrium(im12.Peqhat, csi.Phat, csi.etild, im12.alp,
                                         csi.alp, im12.gdd, csi.gdd, csi.guu);
    if (Peqhatm1 < 0.0) {
      // Abort the equilibrium calculation if there is no equilibrium.
      return;
    }
    // Store these values now so that the compiler can clear much larger objects from
    // register space.
    Real Peqhatm12 = im12.Peqhat;
    Real dPhatm1 = csi.Phat - Peqhatm1;

    Real dPhatm2;
    // Only continue integrating if we're not using PLM.
    if constexpr (recon != ReconstructionMethod::plm) {
      // Integrate to the i-3/2 interface
      WBStateInterface im32{InterfacePolicy<ivx,2,-2>(), adm, csi, Peqhatm1, m, k, j, i};
      if (im32.Peqhat < 0.0) {
        return;
      }

      // Integrate to the i-2 cell
      csi.SetState(pseos, adm, temp, w0, nscal, m, k-2*dk, j-2*dj, i-2*di);
      Real Peqhatm2 = IntegrateEquilibrium(im32.Peqhat, csi.Phat, csi.etild, im32.alp,
                                           csi.alp, im32.gdd, csi.gdd, csi.guu);
      if (Peqhatm2 < 0.0) {
        // Abort the equilibrium calculation if there is no equilibrium.
        return;
      }
      dPhatm2 = csi.Phat - Peqhatm2;
    }

    Real Phatl, Phatr;
    if constexpr (recon == ReconstructionMethod::plm) {
      PLM(dPhatm1, 0, dPhatp1, Phatl, Phatr);
    } else if constexpr (recon == ReconstructionMethod::ppm4) {
      PPM4(dPhatm2, dPhatm1, 0, dPhatp1, dPhatp2, Phatl, Phatr);
    } else if constexpr (recon == ReconstructionMethod::ppmx) {
      PPMX(dPhatm2, dPhatm1, 0, dPhatp1, dPhatp2, Phatl, Phatr);
    } else if constexpr (recon == ReconstructionMethod::wenoz) {
      WENOZ(dPhatm2, dPhatm1, 0, dPhatp1, dPhatp2, Phatl, Phatr);
    } else if constexpr (recon == ReconstructionMethod::teno) {
      TENO(dPhatm2, dPhatm1, 0, dPhatp1, dPhatp2, Phatl, Phatr);
    }
    wl(m, IPR, k + dk, j + dj, i + di) = (Phatl + Peqhatp12)/volp12;
    wr(m, IPR, k, j, i) = (Phatr + Peqhatm12)/volm12;
  }
}

template<int ivx, class EOSPolicy, class ErrorPolicy>
inline void WellBalancedDispatch(ReconstructionMethod recon, const char *name,
                        int nmb1, int kl, int ku, int jl, int ju, int il, int iu,
                        const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& pseos,
                        const DvceArray5D<Real> temp, const adm::ADM::ADM_vars& adm,
                        const DvceArray5D<Real> w0,
                        const DvceArray5D<Real> wl, const DvceArray5D<Real> wr,
                        const int nscal) {
  switch (recon) {
    case ReconstructionMethod::plm:
      par_for(name, DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        WellBalancedCellT<ReconstructionMethod::plm, ivx>(
            m, k, j, i, pseos, temp, adm, w0, wl, wr, nscal);
      });
      break;
    case ReconstructionMethod::ppm4:
      par_for(name, DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        WellBalancedCellT<ReconstructionMethod::ppm4, ivx>(
            m, k, j, i, pseos, temp, adm, w0, wl, wr, nscal);
      });
      break;
    case ReconstructionMethod::ppmx:
      par_for(name, DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        WellBalancedCellT<ReconstructionMethod::ppmx, ivx>(
            m, k, j, i, pseos, temp, adm, w0, wl, wr, nscal);
      });
      break;
    case ReconstructionMethod::wenoz:
      par_for(name, DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        WellBalancedCellT<ReconstructionMethod::wenoz, ivx>(
            m, k, j, i, pseos, temp, adm, w0, wl, wr, nscal);
      });
      break;
    case ReconstructionMethod::teno:
      par_for(name, DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        WellBalancedCellT<ReconstructionMethod::teno, ivx>(
            m, k, j, i, pseos, temp, adm, w0, wl, wr, nscal);
      });
      break;
    case ReconstructionMethod::dc:
    default:
      par_for(name, DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        WellBalancedCellT<ReconstructionMethod::dc, ivx>(
            m, k, j, i, pseos, temp, adm, w0, wl, wr, nscal);
      });
      break;
  }
}
#endif
