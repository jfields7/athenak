//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file well_balancing.cpp
//! \brief Unit test to ensure that DynGRMHD well-balancing operators are consistent

#include <limits>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "dyn_grmhd/dyn_grmhd_wb.hpp"
#include "utils/finite_diff.hpp"


//----------------------------------------------------------------------------------------
//! \fn template<int nghosts> bool CheckBalance
//! \brief Checks that the flux balances the source properly
template<int nghost>
bool CheckBalance(MeshBlockPack* pmbp, Real e0, Real tol) {
  // Capture variables for kernel
  auto& w0_ = pmbp->pmhd->w0;
  auto& adm_ = pmbp->padm->adm;
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pmbp->pmesh->mb_indcs;
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


  bool global_success = true;
  int nx1 = indcs.nx1;
  int nx2 = indcs.nx2;
  int nx3 = indcs.nx3;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  Kokkos::parallel_reduce("pgen_test", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, bool &success) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real &dx = size.d_view(m).dx1;
    Real &dy = size.d_view(m).dx2;
    Real &dz = size.d_view(m).dx3;
    Real invdx[3] = {1.0/dx, 1.0/dy, 1.0/dz};

    // Compute the source term
    Real dalp = Dxvol<nghost>(0, invdx, adm_.alpha, m, k, j, i);
    // Note that getting machine precision requires dalp to be computed the same way as
    // as it shows up via (P_{i+1/2} - P_{i-1/2})/\Delta x. This is not computationally
    // efficient because it requires computing separate left and right states and
    // subtracting them rather than a single combined calculation. For the time being,
    // a single combined calculation is done because the error is still O(10^-14).
    /*Real dalp = (0.5*(adm_.alpha(m, k, j, i+1) + adm_.alpha(m, k, j, i)) -
                0.5*(adm_.alpha(m, k, j, i-1) + adm_.alpha(m, k, j, i)))/dx;*/
    Real source = -e0*dalp;

    // Compute the equilibrium pressures; note that we assume a constant metric.
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> gdd;
    const Real& alpha = adm_.alpha(m, k, j, i);
    gdd(0, 0) = gdd(1, 1) = gdd(2, 2) = 1.0;
    gdd(0, 1) = gdd(0, 2) = gdd(1, 2) = 0.0;
    Real alpp = InterpToInterface<IVX, nghost, 1>(adm_.alpha, k, j, i, m);
    Real alpm = InterpToInterface<IVX, nghost, -1>(adm_.alpha, k, j, i, m);
    Real Phat0 = w0_(m, IPR, k, j, i)*alpha;

    Real Peqp = IntegrateEquilibrium(Phat0, Phat0, e0, alpha, alpp, gdd, gdd, gdd);
    Real Peqm = IntegrateEquilibrium(Phat0, Phat0, e0, alpha, alpm, gdd, gdd, gdd);

    Real flux = (Peqp - Peqm)/dx;

    Real err = (flux - source)/source;
    if (Kokkos::fabs(err) > tol) {
      Kokkos::printf("The flux and source do not balance!\n"
                     "  source: %20.17g\n"
                     "  flux: %20.17g\n"
                     "  error: %20.17g\n",
                     source, flux, err);
      success = false;
    }
  }, Kokkos::LAnd<bool>(global_success));

  return global_success;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::WellBalancing
//! \brief Runs DynGRMHD well balancing unit tests
//
//  The well-balanced scheme should be exact for a hydrostatic atmosphere with constant
//  energy density and linear gravity. We test that this is the case.

void ProblemGenerator::WellBalancing(ParameterInput *pin, const bool restart) {
  MeshBlockPack* pmbp = pmy_mesh_->pmb_pack;

  // Total energy density
  Real e0 = pin->GetOrAddReal("problem", "e0", 1.0);
  // Rest-mass density at the base of the atmosphere
  Real rho0 = pin->GetOrAddReal("problem", "rho0", 0.8);
  // Lapse at the base of the atmosphere
  Real alp0 = pin->GetOrAddReal("problem", "alp0", 0.4);
  // Gravitational gradient
  Real g = pin->GetOrAddReal("problem", "grav", 0.1);

  if (pmbp->pdyngr == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Well-balancing unit test only works for DynGRMHD!\n";
    exit(EXIT_FAILURE);
  }

  auto& w0_ = pmbp->pmhd->w0;
  auto& adm_ = pmbp->padm->adm;

  // Capture variables for kernel
  auto &size = pmbp->pmb->mb_size;
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
  Real gm1 = pmbp->pmhd->peos->eos_data.gamma - 1.0;

  Real P0 = (e0 - rho0)*gm1;

  // Set the initial data to a hydrostatic atmosphere with linear gravity and constant
  // energy density
  par_for("pgen_setup", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &dx = size.d_view(m).dx1;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    
    Real alpha = alp0 + (x1min + i*dx)*g;
    Real P = P0 - e0*(x1min + i*dx)*g;
    Real rho = e0 - P*gm1;

    adm_.alpha(m, k, j, i) = alpha;
    adm_.g_dd(m, 0, 0, k, j, i) = 1.0;
    adm_.g_dd(m, 1, 1, k, j, i) = 1.0;
    adm_.g_dd(m, 2, 2, k, j, i) = 1.0;
    adm_.g_dd(m, 0, 1, k, j, i) = 0.0;
    adm_.g_dd(m, 0, 2, k, j, i) = 0.0;
    adm_.g_dd(m, 1, 2, k, j, i) = 0.0;

    adm_.vK_dd(m, 0, 0, k, j, i) = 0.0;
    adm_.vK_dd(m, 0, 1, k, j, i) = 0.0;
    adm_.vK_dd(m, 0, 2, k, j, i) = 0.0;
    adm_.vK_dd(m, 1, 1, k, j, i) = 0.0;
    adm_.vK_dd(m, 1, 2, k, j, i) = 0.0;
    adm_.vK_dd(m, 2, 2, k, j, i) = 0.0;

    w0_(m, IDN, k, j, i) = rho;
    w0_(m, IPR, k, j, i) = P;
    w0_(m, IVX, k, j, i) = 0.0;
    w0_(m, IVY, k, j, i) = 0.0;
    w0_(m, IVZ, k, j, i) = 0.0;
  });

  // Now that the data is set, check for consistency; this time, we only loop over
  // the physical data rather than including ghost zones.
  bool global_success = CheckBalance<2>(pmbp, e0, 5e-14);
  global_success &= CheckBalance<3>(pmbp, e0, 5e-14);
  global_success &= CheckBalance<4>(pmbp, e0, 5e-14);

  if (!global_success) {
    std::cout << "The test was not successful...\n";
    exit(EXIT_FAILURE);
  }

  std::cout << "Success!\n";

  return;
}
