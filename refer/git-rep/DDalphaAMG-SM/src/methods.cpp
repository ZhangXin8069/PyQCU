#include "methods.h"

void Methods::BiCG(const int max_it, const bool print){   
    if (mpi::rank2d == 0) 
        std::cout << "--------------Bi-CGstab inversion--------------" << std::endl;
    BiCG::max_iter = max_it;
    BiCG::tol = tol;
    localFLOPS = 0;

    start = MPI_Wtime(); 
    iter_counters::BiCGIt = bi_cgstab(U,rhs,x0,xBiCG,m0,print);
    end = MPI_Wtime(); 

    mpi_reduceFLOPS();
    printFLOPS(FLOPS);
    flop_counters::BiCGFLops=FLOPS;
    if (mpi::rank2d == 0)
        printf("[rank %d] time elapsed Bi-CGstab: %.4fs.\n\n", mpi::rank2d, end - start);
}

void Methods::GMRES(const int len, const int restarts,const bool print){
    if (mpi::rank2d == 0)
        std::cout << "--------------GMRES without preconditioning--------------" << std::endl;
    int x_ini = 1, t_ini = 1, x_fin = mpi::width_x, t_fin = mpi::width_t;
    FGMRES_fine_level fgmres_fl(mpi::width_t*mpi::width_x, LV::dof, mpi::maxSizeH,
        x_ini, t_ini, 
        x_fin, t_fin, len, restarts, tol, U, m0);
    
    localFLOPS = 0;

    start = MPI_Wtime(); 
    iter_counters::GMRESIt = fgmres_fl.fgmres(rhs,x0,xGMRES,print);
    end = MPI_Wtime(); 

    mpi_reduceFLOPS();
    printFLOPS(FLOPS);
    flop_counters::GMRESFlops=FLOPS;
    if (mpi::rank2d == 0)
        printf("[rank %d] time elapsed GMRES: %.4fs.\n\n", mpi::rank2d, end - start);
}

void Methods::CG(const bool print){
    if (mpi::rank2d == 0)
        std::cout << "--------Inverting the normal equations with CG----------" << std::endl; 

    localFLOPS = 0;

    spinor D_dagg_psi(mpi::maxSizeH);
    D_dagger_phi(U, rhs, D_dagg_psi, m0);
    start = MPI_Wtime();
    iter_counters::CGIt = conjugate_gradient(U, D_dagg_psi, xCG, m0,print);
    end = MPI_Wtime();

    mpi_reduceFLOPS();
    printFLOPS(FLOPS);
    flop_counters::CGFlops=FLOPS;
    if (mpi::rank2d == 0)
        printf("[rank %d] time elapsed CG: %.4fs.\n\n", mpi::rank2d, end - start);  
}

void Methods::SAP(const int iterations, const int xblocks, const int tblocks,const bool print){
    double tol = 1e-10;
    if (mpi::rank2d == 0)
        std::cout << "--------------SAP as stand-alone solver --------------" << std::endl;
    SAP_fine_level sap(mpi::width_x,  mpi::width_t, xblocks, tblocks, 2, 1);
    sap.set_params(U,mass::m0);

    localFLOPS = 0;

    start = MPI_Wtime();
    iter_counters::SAPIt = sap.SAP(rhs,xSAP,iterations,tol,print);
    end = MPI_Wtime();

    mpi_reduceFLOPS();
    printFLOPS(FLOPS);
    flop_counters::SAPFlops=FLOPS;
    if (mpi::rank2d == 0)
        printf("[rank %d] time elapsed SAP: %.4fs.\n\n", mpi::rank2d, end - start);
}



void Methods::FGMRES_sap(const int len, const int restarts, const bool print){
    if (mpi::rank2d == 0)
        std::cout << "--------------Flexible GMRES with SAP preconditioning--------------" << std::endl;
    
    int x_ini = 1, t_ini = 1, x_fin = mpi::width_x, t_fin = mpi::width_t;
    FGMRES_SAP fgmres_sap(LV::Ntot, LV::dof, mpi::maxSizeH,
    x_ini, t_ini, 
    x_fin, t_fin, len, restarts, tol, U, m0); 

    localFLOPS = 0;

    start = MPI_Wtime();
    iter_counters::FGMRES_SAPIt = fgmres_sap.fgmres(rhs,x0, xFGMRES_SAP,print);
    end = MPI_Wtime();

    mpi_reduceFLOPS();
    printFLOPS(FLOPS);
    flop_counters::FGMRES_SAPFlops=FLOPS;

    if (mpi::rank2d == 0)
        printf("[rank %d] time elapsed FGMRES_SAP: %.4fs.\n\n", mpi::rank2d, end - start);
}

void Methods::Vcycle(const int iterations,const bool print){
    if (mpi::rank2d == 0)
        std::cout << "--------------Stand-alone AMG with a V-cycle----------------" << std::endl;

    AlgebraicMG AMG(U, m0,AMGV::nu1, AMGV::nu2);
    AMGV::cycle == 0; //V-cycle
    start = MPI_Wtime();
    AMG.setUpPhase(AMGV::Nit);
    AMG.applyMultilevel(iterations, rhs, xVcycle,tol,print);  
    end = MPI_Wtime();
    if (mpi::rank2d == 0)
        printf("[rank %d] time elapsed V-cycle AMG: %.4fs.\n\n", mpi::rank2d, end - start);
}

void Methods::Kcycle(const int iterations,const bool print){
    if (mpi::rank2d == 0)
        std::cout << "--------------Stand-alone AMG with a K-cycle----------------" << std::endl;
    AlgebraicMG AMG(U, m0,AMGV::nu1, AMGV::nu2);
    AMGV::cycle = 1; //K-cycle
    start = MPI_Wtime();
    AMG.setUpPhase(AMGV::Nit);
    AMG.applyMultilevel(iterations, rhs, xKcycle,tol,print);  
    end = MPI_Wtime();
    if (mpi::rank2d == 0)
        printf("[rank %d] time elapsed K-cycle AMG: %.4fs.\n\n", mpi::rank2d, end - start);
}

void Methods::FGMRES_amg_kcycle(const int nu1, const int nu2,const bool print){
    if (mpi::rank2d == 0)
        std::cout << "--------------FGMRES with AMG K-cycle--------------" << std::endl;

    localFLOPS = 0;
    start = MPI_Wtime();
    FGMRES_AMG_k_cycle f_amg(U, FGMRESV::fgmres_restart_length, FGMRESV::fgmres_restarts, tol,nu1, nu2,m0);
    flop_counters::kCycleSetUpFlops=FLOPS;
    iter_counters::kCycleIt = f_amg.fgmres(rhs,x0,xFGMRES_AMG_kcycle,print);
    end = MPI_Wtime();
    
    mpi_reduceFLOPS();
    printFLOPS(FLOPS);
    flop_counters::kCycleFlops=FLOPS;

    if (mpi::rank2d == 0)
        printf("[rank %d] time elapsed FGMRES AMG K-cycle: %.4fs.\n\n", mpi::rank2d, end - start);
}


void Methods::FGMRES_amg_vcycle(const int nu1, const int nu2,const bool print){
    if (mpi::rank2d == 0)
        std::cout << "--------------FGMRES with AMG V-cycle--------------" << std::endl;

    localFLOPS = 0; 
    
    start = MPI_Wtime();
    FGMRES_AMG_v_cycle f_amg(U, FGMRESV::fgmres_restart_length, FGMRESV::fgmres_restarts, tol,nu1, nu2,m0);
    flop_counters::vCycleSetUpFlops=FLOPS;
    iter_counters::vCycleIt = f_amg.fgmres(rhs,x0,xFGMRES_AMG_vcycle,print);
    end = MPI_Wtime();

    mpi_reduceFLOPS();
    printFLOPS(FLOPS);
    flop_counters::vCycleFlops=FLOPS;
    if (mpi::rank2d == 0)
        printf("[rank %d] time elapsed FGMRES AMG V-cycle: %.4fs.\n\n", mpi::rank2d, end - start);
}


void Methods::check_solution(const spinor& x_sol){
    spinor x(mpi::maxSizeH);
    D_phi(U,x_sol,x,m0);
    int index;
    for (int nx=1;nx<=mpi::width_x;nx++){
    for(int nt=1;nt<=mpi::width_t;nt++){
    for(int mu=0; mu<2; mu++){
        index = idx(nx,nt,mu);
        if (std::abs(x.val[index]-rhs.val[index]) > 1e-6 ){
            std::cout << "Solution differs at nx" << nx << " nt " << nt << " mu " << mu << " rank" << mpi::rank2d << std::endl;
            std::cout << x.val[index] << "   " << rhs.val[index] << std::endl;
        }
    }
    }
    }

    if (mpi::rank2d == 0)
        std::cout << "Solution verified" << std::endl;
}