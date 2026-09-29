#ifndef UTILS_LORENTZ_BOOST_HPP_
#define UTILS_LORENTZ_BOOST_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lorentz_boost.hpp
//! \brief Utilities for Lorentz-boosting spacetime variables.

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "athena_tensor.hpp"

namespace lorentz {

// This computes a generic boost \Lambda_{\nu'}^\mu along a four-velocity.
KOKKOS_INLINE_FUNCTION
void ComputeBoost(AthenaPointTensor<Real, TensorSymm::NONE, 4, 2>& l_du,
                  const alpha,
                  const AthenaPointTensor<Real, TensorSymm::NONE, 3, 1>& beta_u,
                  const AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2>& g_dd,
                  const AthenaPointTensor<Real, TensorSymm::NONE, 3, 1>& Wv_u) {
  // Compute the lowered velocity and the Lorentz factor.
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1>& Wv_d;
  Real W = 0;
  for (int a = 0; a < 3; a++) {
    Wv_d(a) = 0.;
    for (int b = 0; b < 3; b++) {
      Wv_d(a) += g_dd(a, b)*Wv_u(b);
    }
    W += Wv_d(a)*Wv_u(a);
  }
  W = Kokkos::sqrt(1. + W);

  Real ialpha = 1.0/alpha;
  Real iden = 1.0/(1. + W);

  l_du(0, 0) = W*ialpha;
  l_du(0, 1) = Wv_d(0)*ialpha;
  l_du(0, 2) = Wv_d(1)*ialpha;
  l_du(0, 3) = Wv_d(2)*ialpha;

  l_du(1, 0) = Wv_u(0) - W*beta_u(0)*ialpha;
  l_du(1, 1) = 1. + Wv_u(0)*Wv_d(0)*iden - beta_u(0)*Wv_d(0)*ialpha;
  l_du(1, 2) = Wv_u(1)*Wv_d(0)*iden - beta_u(1)*Wv_d(0)*ialpha;
  l_du(1, 3) = Wv_u(2)*Wv_d(0)*iden - beta_u(2)*Wv_d(0)*ialpha;

  l_du(2, 0) = Wv_u(1) - W*beta_u(1)*ialpha;
  l_du(2, 1) = Wv_u(0)*Wv_d(1)*iden - beta_u(0)*Wv_d(1)*ialpha;
  l_du(2, 2) = 1. + Wv_u(1)*Wv_d(1)*iden - beta_u(1)*Wv_d(1)*ialpha;
  l_du(2, 3) = Wv_u(2)*Wv_d(1)*iden - beta_u(2)*Wv_d(1)*ialpha;

  l_du(3, 0) = Wv_u(2) - W*beta_u(2)*ialpha;
  l_du(3, 1) = Wv_u(0)*Wv_d(2)*iden - beta_u(0)*Wv_d(2)*ialpha;
  l_du(3, 2) = Wv_u(1)*Wv_d(2)*iden - beta_u(1)*Wv_d(2)*ialpha;
  l_du(3, 3) = 1. + Wv_u(2)*Wv_d(2)*iden - beta_u(2)*Wv_d(2)*ialpha;
}

KOKKOS_INLINE_FUNCTION
void BoostSpacetimeMetric(AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2>& gp_dd,
                          const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2>& g_dd,
                          const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2>& l_du) {
  for (int a = 0; a < 4; a++) {
    for (int b = a; b < 4; b++) {
      gp_dd(a, b) = 0;
      for (int c = 0; c < 4; c++) {
        for (int d = 0; d < 4; d++) {
          gp_dd(a, b) += g_dd(c, d)*l_du(a, c)*l_du(b, d);
        }
      }
    }
  }
}

} // namespace lorentz

#endif // UTILS_LORENTZ_BOOST_HPP_
