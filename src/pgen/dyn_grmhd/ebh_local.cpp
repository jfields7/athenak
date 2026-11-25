//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyngr_tov.cpp
//  \brief Problem generator for TOV star. Only works when ADM is enabled.

#include <math.h>     // abs(), cos(), exp(), log(), NAN, pow(), sin(), sqrt()

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "utils/tov/tov.hpp"
#include "utils/tov/tov_polytrope.hpp"
#include "utils/tov/tov_tabulated.hpp"
#include "utils/tov/tov_piecewise_poly.hpp"

#include <Kokkos_Random.hpp>

template<class TOVEOS>
void SetupPuncture(ParameterInput *pin, Mesh* pmy_mesh_) {
  Real rho = pin->GetOrAddReal("problem", "rhoc", 1.28e-3);
  Real mass = pin->GetOrAddReal("problem", "mass", 1.0e-3);
  Real lapse = pin->GetOrAddReal("problem", "lapse", 1.0);

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // Use the TOV solver with the specified EOS.
  TOVEOS eos{pin};

  Real P = eos.template GetPFromRho<tov::LocationTag::Host>(rho);
  Real Ye = 0.0;
  constexpr bool use_ye = tov::UsesYe<TOVEOS>;
  if constexpr (use_ye) {
    Ye = eos.template GetYeFromRho<tov::LocationTag::Host>(rho);
  }


  //auto& u0_ = pmbp->pmhd->u0;
  auto& w0_ = pmbp->pmhd->w0;
  int& nvars_ = pmbp->pmhd->nmhd;
  int& nscal_ = pmbp->pmhd->nscalars;

  // Capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  int &ie = indcs.ie;
  int &je = indcs.je;
  int &ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;

  auto &size = pmbp->pmb->mb_size;
  auto &adm = pmbp->padm->adm;
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  par_for("pgen_tov1", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    // Set hydrodynamic quantities
    w0_(m,IDN,k,j,i) = rho;
    w0_(m,IPR,k,j,i) = P;
    w0_(m,IVX,k,j,i) = 0.0;
    w0_(m,IVY,k,j,i) = 0.0;
    w0_(m,IVZ,k,j,i) = 0.0;
    auto &nvars = nvars_;
    auto &nscal = nscal_;
    if (use_ye && nscal >= 1) {
      w0_(m,nvars,k,j,i) = Ye;
    }

    // Use Schwarzschild in KS coordinates
    // Need radial coordinate to find psi4
    Real r = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
    const Real eps = 1e-3*mass;
    if (r < eps) {
      r = 0.5*(eps + r*r/eps);
    }
    Real l[3] = {x1v/r, x2v/r, x3v/r};
    Real H = mass/r;
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        adm.g_dd(m,a,b,k,j,i) = 2*H*l[a]*l[b];
      }
    }
    adm.g_dd(m,0,0,k,j,i) += 1.0;
    adm.g_dd(m,1,1,k,j,i) += 1.0;
    adm.g_dd(m,2,2,k,j,i) += 1.0;

    // We rescale the lapse based on the expected lapse of the center of the star.
    Real alpha = lapse/Kokkos::sqrt(1.0 + 2*H);
    adm.alpha(m,k,j,i) = alpha;

    for (int a = 0; a < 3; a++) {
      adm.beta_u(m,a,k,j,i) = 2.*lapse*H/(1. + 2*H)*l[a];
    }

    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        adm.vK_dd(m,a,b,k,j,i) = 2*H/(r*Kokkos::sqrt(1. + 2*H))*(((a == b) ? 1.0 : 0.0) -
                                                (2. + H)*l[a]*l[b]);
      }
    }
  });

  // parse some parameters
  Real b_norm = pin->GetOrAddReal("problem", "b_norm", 0.0);

  // compute magnetic field for all faces
  int nmb = pmbp->nmb_thispack;
  auto &b0 = pmbp->pmhd->b0;
  par_for("pgen_Bfc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    b0.x1f(m,k,j,i) = 0.0;
    b0.x2f(m,k,j,i) = 0.0;
    b0.x3f(m,k,j,i) = b_norm;

    // Include extra face-component at edge of block in each direction
    if (i==ie) {
      b0.x1f(m,k,j,i+1) = 0.0;
    }
    if (j==je) {
      b0.x2f(m,k,j+1,i) = 0.0;
    }
    if (k==ke) {
      b0.x3f(m,k+1,j,i) = b_norm;
    }
  });

  // Compute cell-centered fields
  auto &bcc_ = pmbp->pmhd->bcc0;
  par_for("pgen_Bcc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // cell-centered fields are simple linear average of face-centered fields
    Real& w_bx = bcc_(m,IBX,k,j,i);
    Real& w_by = bcc_(m,IBY,k,j,i);
    Real& w_bz = bcc_(m,IBZ,k,j,i);
    w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
    w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
    w_bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
  });
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//  \brief Sets initial conditions for TOV star in DynGRMHD
//  Compile with '-D PROBLEM=dyngr_tov' to enroll as user-specific problem generator

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "TOV star problem can only be run when <adm> block is present"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // Select the right TOV template based on the EOS we need.
  if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_ideal) {
    SetupPuncture<tov::PolytropeEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_compose) {
    SetupPuncture<tov::TabulatedEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_hybrid) {
    SetupPuncture<tov::TabulatedEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_piecewise_poly) {
    SetupPuncture<tov::PiecewisePolytropeEOS>(pin, pmy_mesh_);
  } else {
    std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Unknown EOS requested for TOV star problem" << std::endl
              << "Defaulting to fixed polytropic EOS" << std::endl;
    SetupPuncture<tov::PolytropeEOS>(pin, pmy_mesh_);
  }

  // Mesh block info for loop limits
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;

  pmbp->pcoord->UpdateExcisionMasks();

  // Convert primitives to conserved
  pmbp->pdyngr->PrimToConInit(0, (n1-1), 0, (n2-1), 0, (n3-1));

  if (pmbp->pz4c != nullptr) {
    switch (indcs.ng) {
      case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
              pmbp->pz4c->ADMConstraints<2>(pmbp);
              break;
      case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
              pmbp->pz4c->ADMConstraints<3>(pmbp);
              break;
      case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
              pmbp->pz4c->ADMConstraints<4>(pmbp);
              break;
    }
  }

  return;
}
