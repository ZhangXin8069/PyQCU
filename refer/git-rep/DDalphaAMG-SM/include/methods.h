#ifndef METHODS_H
#define METHODS_H

#include "amg.h"
#include "conjugate_gradient.h"

//Class for comparing different matrix-inversion methods.
class Methods {
public:
    Methods(const spinor& U, const spinor& rhs, const spinor& x0 ,const double m0, const double tol): 
    U(U), rhs(rhs), x0(x0), m0(m0),tol(tol){
        //Solution buffers
        xBiCG       = spinor(mpi::maxSizeH);
        xCG         = spinor(mpi::maxSizeH);
        xGMRES      = spinor(mpi::maxSizeH);
        xSAP        = spinor(mpi::maxSizeH);
        xFGMRES_AMG_kcycle = spinor(mpi::maxSizeH);
        xFGMRES_AMG_vcycle = spinor(mpi::maxSizeH);
        xFGMRES_SAP = spinor(mpi::maxSizeH);
        xVcycle     = spinor(mpi::maxSizeH);
        xKcycle     = spinor(mpi::maxSizeH);
    }
    ~Methods(){}

    void BiCG(const int max_it,const bool print);
    void GMRES(const int len, const int restarts,const bool print);
    void CG(const bool print);
    void SAP(const int iterations,const int xblocks, const int tblocks,const bool print);
    void FGMRES_sap(const int len, const int restarts,const bool print);
    
    void FGMRES_amg_kcycle(const int nu1, const int nu2,const bool print);
    void FGMRES_amg_vcycle(const int nu1, const int nu2,const bool print);
    void Vcycle(const int iterations,const bool print);
    void Kcycle(const int iterations,const bool print);

    void check_solution(const spinor& x_sol);

    spinor xBiCG;
    spinor xCG;
    spinor xGMRES;
    spinor xSAP;
    spinor xFGMRES_AMG_kcycle;
    spinor xFGMRES_AMG_vcycle;
    spinor xFGMRES_SAP;
    spinor xVcycle;
    spinor xKcycle;

private:

    const spinor U;
    const spinor rhs;
    const spinor x0;
    const double m0;
    const double tol;
    double start, end;

};


#endif