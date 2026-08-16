#ifndef DIRAC_OPERATOR_INCLUDED
#define DIRAC_OPERATOR_INCLUDED
#include "boundary.h"
#include "halo_exchange.h"
#include "utils.h"



/*
	Dirac operator application D phi
	U: gauge configuration
	phi: spinor to apply the operator to
	m0: mass parameter
*/
void D_phi(const spinor& U, const spinor& phi, spinor& Dphi, const double& m0);


/*
	Dirac dagger operator application D^+ phi
	U: gauge configuration
	phi: spinor to apply the operator to
	m0: mass parameter
*/
void D_dagger_phi(const spinor& U, const spinor& phi, spinor& Dphi, const double& m0);


/*
	Application of D^+ D
	It calls the previous functions
*/
void D_D_dagger_phi(const spinor& U, const spinor& phi, spinor& Dphi, const double& m0);





#endif