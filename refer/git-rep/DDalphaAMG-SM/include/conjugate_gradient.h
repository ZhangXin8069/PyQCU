#ifndef CONJUGATE_GRADIENT_H
#define CONJUGATE_GRADIENT_H

#include "dirac_operator.h"

/*
    Conjugate gradient method for computing (DD^dagger)^-1 phi 
    U: gauge configuration
    phi: right-hand side vector
    m0: mass parameter for Dirac matrix 
        
    The convergence criterion is ||r|| < ||phi|| * tol
*/
int conjugate_gradient(const spinor& U, const spinor& phi, spinor& x, const double& m0,const bool print); 

int bi_cgstab(const spinor& U, const spinor& phi, const spinor& x0, spinor& x, const double& m0, const bool& print_message);

#endif
