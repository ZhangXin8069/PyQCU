#include "tests.h"

void test_gather_Datatypes_level_class(Level& lev,const int dof){
    if (mpi::size != 16){
         printf("This test is meant to be run with 16 processes.\n");
        MPI_Abort(mpi::cart_comm, EXIT_FAILURE);
    }
    
    spinor input((lev.Nt+2)*(lev.Nx+2)*dof);
    int n;
    for(int x = 1; x<=lev.Nx; x++){
        for(int t = 1; t<=lev.Nt; t++){
            n = x*(lev.Nt+2)+t;
            for(int mu=0; mu<dof; mu++){
                input.val[dof*n+mu] = dof*n+mu + mpi::rank2d;
            }        
        }
    }


    for(int i = 0; i < mpi::size; i++) {
        MPI_Barrier(mpi::cart_comm);
        if (i == mpi::rank2d) {
            std::cout << "rank " << mpi::rank2d << std::endl;
            for(int x = 1; x<=lev.Nx; x++){
                for(int t = 1; t<=lev.Nt; t++){
                    int n = x*(lev.Nt+2) + t;
                    std::cout << "[";
                    for(int mu = 0; mu<dof; mu++){
                         std::cout << input.val[dof*n+mu] << ", ";
                    }   
                    std::cout << "], ";
                }
                std::cout << std::endl;
            } 
        }
    }

  
    spinor buffer((lev.Nx_coarse_rank+2)*(lev.Nt_coarse_rank+2)*dof);

    lev.gather_to_coarse_rank(input, buffer,dof);

   
    int local_rank;
    int commID = mpi::rank_dictionary[mpi::rank2d];  
    MPI_Comm_rank(mpi::coarse_comm[commID], &local_rank);
    for(int i = 0; i <mpi::size; i++) {
        MPI_Barrier(mpi::cart_comm);
        if (i == mpi::rank2d && local_rank == 0) {
            std::cout << "\nGather in rank " << mpi::rank2d << std::endl;
            for(int x = 0; x<lev.Nx_coarse_rank+2; x++){
                for(int t = 0; t<lev.Nt_coarse_rank+2; t++){
                    int n = x*(lev.Nt_coarse_rank+2) + t;
                    std::cout << "[";
                    for(int mu = 0; mu<dof; mu++){
                         std::cout << buffer.val[dof*n+mu] << ", ";
                    }   
                    std::cout << "], ";
                }
                std::cout << std::endl;
            } 
        }
    }
}

void gather_tests(){
    spinor U(mpi::maxSizeH);
    int l = 0;
    Level lev(l,U);
    int DOF = lev.DOF;
    if (mpi::rank2d == 0)
        std::cout << "Test gather for spinor at level " << l << " with DOF=" << DOF << std::endl;
    test_gather_Datatypes_level_class(lev,DOF);
     if (mpi::rank2d == 0)
        std::cout << "Test gather for G1 at level " << l << " with DOF=" << DOF*DOF << std::endl;
    test_gather_Datatypes_level_class(lev,DOF*DOF);
     if (mpi::rank2d == 0)
        std::cout << "Test gather for G2G3 at level " << l << " with DOF=" << DOF*DOF*2 << std::endl;
    test_gather_Datatypes_level_class(lev,DOF*DOF*2);
}


void test_scatter_Datatypes_level_class(Level& lev,const int dof){
    if (mpi::size != 16){
         printf("This test is meant to be run with 16 processes.\n");
        MPI_Abort(mpi::cart_comm, EXIT_FAILURE);
    }
    spinor input((lev.Nx_coarse_rank+2)*(lev.Nt_coarse_rank+2)*dof);
    spinor buffer((lev.Nt+2)*(lev.Nx+2)*dof);

	int commID = mpi::rank_dictionary[mpi::rank2d];
    int local_rank;
    MPI_Comm_rank(mpi::coarse_comm[commID], &local_rank);

    int n;
    for(int x = 1; x<=lev.Nx_coarse_rank; x++){
        for(int t = 1; t<=lev.Nt_coarse_rank; t++){
            n = x*(lev.Nt_coarse_rank+2)+t;
            for(int mu=0; mu<dof; mu++){
                input.val[dof*n+mu] = dof*n+mu + mpi::rank2d;
            }        
        }
    }


    for(int i = 0; i < mpi::size; i++) {
        MPI_Barrier(mpi::cart_comm);
        if (i == mpi::rank2d && local_rank == 0) {
            std::cout << "rank " << mpi::rank2d << std::endl;
            for(int x = 1; x<=lev.Nx_coarse_rank; x++){
                for(int t = 1; t<=lev.Nt_coarse_rank; t++){
                    int n = x*(lev.Nt_coarse_rank+2) + t;
                     std::cout << "[";
                    for(int mu=0; mu<dof;mu++){
                        std::cout << input.val[dof*n+mu] << ", ";
                    }
                    std::cout << "], ";
                }
                std::cout << std::endl;
            } 
        }
    }

    lev.scatter_to_local_rank_from_coarse_rank(input, buffer,dof);
    
    for(int i = 0; i <mpi::size; i++) {
        MPI_Barrier(mpi::cart_comm);
        if (i == mpi::rank2d) {
            std::cout << "\nScatter in rank " << mpi::rank2d << std::endl;
            for(int x = 0; x<lev.Nx+2; x++){
                for(int t = 0; t<lev.Nt+2; t++){
                    int n = x*(lev.Nt+2) + t;
                    std::cout << "[";
                    for(int mu=0; mu<dof;mu++){
                        std::cout << buffer.val[dof*n+mu] << ", ";
                    }
                    std::cout << "], ";
                }
                std::cout << std::endl;
            } 
        }
    }
}

void scatter_tests(){
    spinor U(mpi::maxSizeH);
    int l = 0;
    Level lev(l,U);
    int DOF = lev.DOF;
    if (mpi::rank2d == 0)
        std::cout << "Test scatter for spinor at level " << l << " with DOF=" << DOF << std::endl;
    test_scatter_Datatypes_level_class(lev,DOF);
     if (mpi::rank2d == 0)
        std::cout << "Test scatter for G1 at level " << l << " with DOF=" << DOF*DOF << std::endl;
    test_scatter_Datatypes_level_class(lev,DOF*DOF);
     if (mpi::rank2d == 0)
        std::cout << "Test scatter for G2G3 at level " << l << " with DOF=" << DOF*DOF*2 << std::endl;
    test_scatter_Datatypes_level_class(lev,DOF*DOF*2);
}


void test_Dc_with_rank_coarsening(){
    std::vector<Level*> levels;
    spinor U(mpi::maxSizeH);

    //if (mpi::size != 16){
    //    printf("This test is meant to be run with 4 processes.\n");
    //    MPI_Abort(mpi::cart_comm, EXIT_FAILURE);
    //}
    for(int x = 1; x<=mpi::width_x; x++){
        for(int t = 1; t<=mpi::width_t; t++){
            int n = x*(mpi::width_t+2)+t;
            U.val[2*n]        = RandomU1();
            U.val[2*n+1]      = RandomU1();
        }
    }

    for(int l = 0; l<LevelV::levels; l++){
        Level* level = new Level(l,U);
        levels.push_back(level);
    }


    //Random test vectors ... 
    static std::mt19937 randomInt(mpi::rank2d); 
	std::uniform_real_distribution<double> distribution(-1.0, 1.0); //mu, standard deviation
    for(int l=0; l<LevelV::maxLevel; l++){
        for(int cc = 0; cc < levels[l]->Ntest; cc++){
            for(int x=1; x<=levels[l]->Nx; x++){
            for(int t=1; t<=levels[l]->Nt; t++){
	        for(int c=0; c<levels[l]->colors; c++){
	        for(int s=0; s<2; s++){
                int indx 	= x*(levels[l]->Nt+2)*levels[l]->colors*2 + t*levels[l]->colors*2 + c*2 	+ s;
                levels[l]->tvec[cc].val[indx] = distribution(randomInt) + I_number * distribution(randomInt);            
            }
            }
            }
            }  
        }   
        levels[l]->orthonormalize();                //Orthonormalize test vectors
        levels[l]->checkOrthogonality();            //Checking orthogonality   
        levels[l]->makeCoarseLinks(*levels[l+1]);        //Make coarse links
        if (mpi::rank2d  == 0)
            std::cout << "Proceed to level " << l+1 << std::endl;
    }

    
    
    for(int l = 0; l<LevelV::maxLevel; l++){
        if (mpi::rank2d == 0)
            std::cout << "checking level " << l+1 << std::endl;
        spinor vc((levels[l+1]->Nt+2)*(levels[l+1]->Nx+2)*levels[l+1]->DOF);
        spinor v((levels[l]->Nt_coarse_rank+2)*(levels[l]->Nx_coarse_rank+2)*levels[l]->DOF);
        spinor temp((levels[l]->Nt_coarse_rank+2)*(levels[l]->Nx_coarse_rank+2)*levels[l]->DOF);
        for(int x = 1; x<=levels[l+1]->Nx; x++){
        for(int t = 1; t<=levels[l+1]->Nt; t++){
        for(int dof=0; dof<levels[l+1]->DOF; dof++){
            int n = x*(levels[l+1]->Nt+2)+t;
            vc.val[levels[l+1]->DOF*n+dof] = RandomU1();
        }
        }
        }

        spinor out1((levels[l+1]->Nt_coarse_rank+2)*(levels[l+1]->Nx_coarse_rank+2)*levels[l+1]->DOF);
        spinor out2((levels[l]->tblocks_per_coarse_rank+2)*(levels[l]->xblocks_per_coarse_rank+2)*levels[l+1]->DOF);
        // Dc = P^+ D P
        levels[l+1]->D_operator(vc, out1);

        //Explicit application of each operator. Should coincide with D_operator at leve l+1.
        levels[l]->P_vc(vc,v);
        levels[l]->D_operator(v,temp);
        levels[l]->Pdagg_v(temp,out2);
        
        //Only check on the working ranks of the coarse level ...
        if (levels[l+1]->ranks_comm != MPI_COMM_NULL){
            for(int x = 1; x<=levels[l+1]->Nx; x++){
                for(int t = 1; t<=levels[l+1]->Nt; t++){
                    int n = x*(levels[l+1]->Nt+2)+t;
                    for(int dof=0; dof<levels[l+1]->DOF; dof++){
                        if (std::abs(out1.val[levels[l+1]->DOF*n+dof]-out2.val[levels[l+1]->DOF*n+dof]) > 1e-8){
                            std::cout << "Both implementations of D don't coincide, rank " << mpi::rank2d << " level " << l+1 << std::endl;
                            std::cout << "x " << x << " t " << t << " dof " << dof << std::endl;
                            std::cout << "D_operator vc      " <<  out1.val[levels[l+1]->DOF*n+dof] << std::endl;
                            std::cout << "P^+ D P vc         " <<  out2.val[levels[l+1]->DOF*n+dof] << std::endl;
                            //std::cout << "vc                 " <<  vc.val[levels[l+1]->DOF*n+dof] << std::endl;
                            std::cout << std::endl;      
                            //return; 
                        }
                    } 
                }
            }
            //std::cout << "Dc coincides with P^+ D P at level " << l+1  << " on rank " << mpi::rank2d << std::endl;
        }
            
        if (mpi::rank2d == 0)
            std::cout << "if no error was printed then all good" << std::endl;
        
        
    }
    
        

    for (auto ptr : levels) delete ptr;

}


void test_SAP_in_level_0(){   
    std::vector<Level*> levels;
    spinor U(mpi::maxSizeH);

    //if (mpi::size != 16){
    //    printf("This test is meant to be run with 4 processes.\n");
    //    MPI_Abort(mpi::cart_comm, EXIT_FAILURE);
    //}
    for(int x = 1; x<=mpi::width_x; x++){
        for(int t = 1; t<=mpi::width_t; t++){
            int n = x*(mpi::width_t+2)+t;
            U.val[2*n]        = RandomU1();
            U.val[2*n+1]      = RandomU1();
        }
    }

    for(int l = 0; l<2; l++){
        Level* level = new Level(l,U);
        levels.push_back(level);
    }

    //Random test vectors ... 
    static std::mt19937 randomInt(mpi::rank2d); 
	std::uniform_real_distribution<double> distribution(-1.0, 1.0); //mu, standard deviation
    for(int l=0; l<1; l++){
        for(int cc = 0; cc < levels[l]->Ntest; cc++){
            for(int x=1; x<=levels[l]->Nx; x++){
            for(int t=1; t<=levels[l]->Nt; t++){
	        for(int c=0; c<levels[l]->colors; c++){
	        for(int s=0; s<2; s++){
                int indx 	= x*(levels[l]->Nt+2)*levels[l]->colors*2 + t*levels[l]->colors*2 + c*2 	+ s;
                levels[l]->tvec[cc].val[indx] = distribution(randomInt) + I_number * distribution(randomInt);            
            }
            }
            }
            }  
        }   
        levels[l]->orthonormalize();                //Orthonormalize test vectors
        levels[l]->checkOrthogonality();            //Checking orthogonality   
        levels[l]->makeCoarseLinks(*levels[l+1]);   //Make coarse links
    }


    //Checking that sap_l gives the same result as sap_fine_level
    int l = 0;
    spinor rhs((levels[l]->Nt+2)*(levels[l]->Nx+2)*levels[l]->DOF);
    spinor x_level((levels[l]->Nt+2)*(levels[l]->Nx+2)*levels[l]->DOF);

    for(int x=1; x<=levels[l]->Nx; x++){
    for(int t=1; t<=levels[l]->Nt; t++){
	for(int dof=0; dof<levels[l]->DOF; dof++){
        int indx= (x*(levels[l]->Nt+2)+t)*levels[l]->DOF + dof;
        rhs.val[indx] = RandomU1();
    }
    }
    }
    
    spinor x_fine((levels[l]->Nt+2)*(levels[l]->Nx+2)*levels[l]->DOF);
    SAP_fine_level sap(mpi::width_x,  mpi::width_t, LevelV::SAP_Block_x[l], LevelV::SAP_Block_t[l], 2, 1);
    sap.set_params(U,mass::m0);

    double tol=1e-10;
    bool print=true;
    int nu = 100;
   
    if (mpi::rank2d == 0)
        std::cout << "SAP solver inside class Level" << std::endl;
    levels[l]->sap_l->SAP(rhs,x_level,nu,tol,print);

    if (mpi::rank2d == 0)
        std::cout << "Outer SAP solver" << std::endl;
   
    sap.SAP(rhs,x_fine,nu,tol,print);
    
    for(int x=1; x<=levels[l]->Nx; x++){
    for(int t=1; t<=levels[l]->Nt; t++){
	for(int dof=0; dof<levels[l]->DOF; dof++){
        int indx= (x*(levels[l]->Nt+2)+t)*levels[l]->DOF + dof;
         if (std::abs(x_level.val[indx]-x_fine.val[indx]) > 1e-8){
            std::cout << "Different solutions on rank " << mpi::rank2d << " at level " << l << std::endl;
            std::cout << "x_level " << x_level.val[indx] << std::endl;
            std::cout << "x_fine  " << x_fine.val[indx]  << std::endl;
            return ;
         }
    }
    }
    }
 
    std::cout << "Both implementations give the same solution on rank " << mpi::rank2d << " level " << l << std::endl;

}

void test_SAP_in_every_level(){   
    std::vector<Level*> levels;
    spinor U(mpi::maxSizeH);

    if (mass::m0<0 && mpi::rank2d == 0)
        printf("This test might fail if the original matrix is too ill-conditioned.\nTry with a larger mass\n");
    
    for(int x = 1; x<=mpi::width_x; x++){
        for(int t = 1; t<=mpi::width_t; t++){
            int n = x*(mpi::width_t+2)+t;
            U.val[2*n]        = RandomU1();
            U.val[2*n+1]      = RandomU1();
        }
    }

    for(int l = 0; l<LevelV::levels; l++){
        Level* level = new Level(l,U);
        levels.push_back(level);
    }

    //Random test vectors ... 
    static std::mt19937 randomInt(mpi::rank2d); 
	std::uniform_real_distribution<double> distribution(-1.0, 1.0); //mu, standard deviation
    for(int l=0; l<LevelV::maxLevel; l++){
        for(int cc = 0; cc < levels[l]->Ntest; cc++){
            for(int x=1; x<=levels[l]->Nx; x++){
            for(int t=1; t<=levels[l]->Nt; t++){
	        for(int c=0; c<levels[l]->colors; c++){
	        for(int s=0; s<2; s++){
                int indx 	= x*(levels[l]->Nt+2)*levels[l]->colors*2 + t*levels[l]->colors*2 + c*2 	+ s;
                levels[l]->tvec[cc].val[indx] = distribution(randomInt) + I_number * distribution(randomInt);            
            }
            }
            }
            }  
        }   
        levels[l]->orthonormalize();                //Orthonormalize test vectors
        levels[l]->checkOrthogonality();            //Checking orthogonality   
        levels[l]->makeCoarseLinks(*levels[l+1]);   //Make coarse links
    }


    //Checking that sap_l gives the same result as sap_fine_level
    for(int l = 0; l<LevelV::levels; l++){
    //for(int l = 0; l<1; l++){
    if (levels[l]->ranks_comm != MPI_COMM_NULL){
        spinor rhs((levels[l]->Nt+2)*(levels[l]->Nx+2)*levels[l]->DOF);
        spinor x_level((levels[l]->Nt+2)*(levels[l]->Nx+2)*levels[l]->DOF);
        spinor D_x_level((levels[l]->Nt+2)*(levels[l]->Nx+2)*levels[l]->DOF);

        for(int x=1; x<=levels[l]->Nx; x++){
        for(int t=1; t<=levels[l]->Nt; t++){
	    for(int dof=0; dof<levels[l]->DOF; dof++){
            int indx= (x*(levels[l]->Nt+2)+t)*levels[l]->DOF + dof;
            rhs.val[indx] = RandomU1();
        }
        }
        }
    
   
        double tol=1e-10;
        bool print=true;
        int nu = 100;
   
        levels[l]->sap_l->SAP(rhs,x_level,nu,tol,print); //D^-1 rhs
        
        levels[l]->D_operator(x_level,D_x_level);//D D^-1 rhs

        for(int x=1; x<=levels[l]->Nx; x++){
        for(int t=1; t<=levels[l]->Nt; t++){
	    for(int dof=0; dof<levels[l]->DOF; dof++){
            int indx= (x*(levels[l]->Nt+2)+t)*levels[l]->DOF + dof;
            if (std::abs(D_x_level.val[indx]-rhs.val[indx]) > 1e-8){
                std::cout << "rhs /= D_operator (D^-1 rhs) on rank " << mpi::rank2d << " at level " << l << std::endl;
                std::cout << "D_x_level " << D_x_level.val[indx] << std::endl;
                std::cout << "rhs       " << rhs.val[indx]  << std::endl;
                return ;
            }
        }
        }
        }
 
        if (mpi::rank2d == 0)
            std::cout << "D_operator (SAP_l rhs) = rhs on rank " << mpi::rank2d << " level " << l << " i.e. solution is correct" << std::endl;
    }
    }
}


void test_gmres_coarse_level(){
    std::vector<Level*> levels;
    spinor U(mpi::maxSizeH);

    if (mass::m0<0 && mpi::rank2d == 0)
        printf("This test might fail if the original matrix is too ill-conditioned.\nTry with a larger mass\n");
    
    for(int x = 1; x<=mpi::width_x; x++){
        for(int t = 1; t<=mpi::width_t; t++){
            int n = x*(mpi::width_t+2)+t;
            U.val[2*n]        = RandomU1();
            U.val[2*n+1]      = RandomU1();
        }
    }

    for(int l = 0; l<LevelV::levels; l++){
        Level* level = new Level(l,U);
        levels.push_back(level);
    }

    //Random test vectors ... 
    static std::mt19937 randomInt(mpi::rank2d); 
	std::uniform_real_distribution<double> distribution(-1.0, 1.0); //mu, standard deviation
    for(int l=0; l<LevelV::maxLevel; l++){
        for(int cc = 0; cc < levels[l]->Ntest; cc++){
            for(int x=1; x<=levels[l]->Nx; x++){
            for(int t=1; t<=levels[l]->Nt; t++){
	        for(int c=0; c<levels[l]->colors; c++){
	        for(int s=0; s<2; s++){
                int indx 	= x*(levels[l]->Nt+2)*levels[l]->colors*2 + t*levels[l]->colors*2 + c*2 	+ s;
                levels[l]->tvec[cc].val[indx] = distribution(randomInt) + I_number * distribution(randomInt);            
            }
            }
            }
            }  
        }   
        levels[l]->orthonormalize();                //Orthonormalize test vectors
        levels[l]->checkOrthogonality();            //Checking orthogonality   
        levels[l]->makeCoarseLinks(*levels[l+1]);   //Make coarse links
    }


    int l = LevelV::maxLevel;
    if (levels[l]->ranks_comm != MPI_COMM_NULL){
        spinor rhs((levels[l]->Nt+2)*(levels[l]->Nx+2)*levels[l]->DOF);
        spinor x0((levels[l]->Nt+2)*(levels[l]->Nx+2)*levels[l]->DOF);
        spinor xGMRES((levels[l]->Nt+2)*(levels[l]->Nx+2)*levels[l]->DOF);
        spinor D_xGMRES((levels[l]->Nt+2)*(levels[l]->Nx+2)*levels[l]->DOF);

        for(int x=1; x<=levels[l]->Nx; x++){
        for(int t=1; t<=levels[l]->Nt; t++){
	    for(int dof=0; dof<levels[l]->DOF; dof++){
            int indx= (x*(levels[l]->Nt+2)+t)*levels[l]->DOF + dof;
            rhs.val[indx] = RandomU1();
        }
        }
        }
    
        double tol=1e-10;
        bool print=true;
        int nu = 100;
   
        levels[l]->gmres_l->fgmres(rhs, x0, xGMRES, print);
        levels[l]->D_operator(xGMRES,D_xGMRES);//D D^-1 rhs

        for(int x=1; x<=levels[l]->Nx; x++){
        for(int t=1; t<=levels[l]->Nt; t++){
	    for(int dof=0; dof<levels[l]->DOF; dof++){
            int indx= (x*(levels[l]->Nt+2)+t)*levels[l]->DOF + dof;
            if (std::abs(D_xGMRES.val[indx]-rhs.val[indx]) > 1e-8){
                std::cout << "rhs /= D_operator (D^-1 rhs) on rank " << mpi::rank2d << " at level " << l << std::endl;
                std::cout << "D_x_level " << D_xGMRES.val[indx] << std::endl;
                std::cout << "rhs       " << rhs.val[indx]  << std::endl;
                return ;
            }
        }
        }
        }
 
        if (mpi::rank2d == 0)
            std::cout << "D_operator (SAP_l rhs) = rhs on rank " << mpi::rank2d << " level " << l << " i.e. solution is correct" << std::endl;
    
    
    }

}

void test_AMG(){
    spinor U(mpi::maxSizeH);
     for(int x = 1; x<=mpi::width_x; x++){
        for(int t = 1; t<=mpi::width_t; t++){
            int n = x*(mpi::width_t+2)+t;
            U.val[2*n]        = RandomU1();
            U.val[2*n+1]      = RandomU1();
        }
    }

    int nu1 = 2;
    int nu2 = 2;
    int Nit = 1;
    AlgebraicMG AMG(U, mass::m0, nu1, nu2); 
    AMG.setUpPhase(Nit);
    AMG.testSetUp();
    AMG.testSAP();
}

void test_open_conf(){

    spinor U(mpi::maxSizeH);
    std::ostringstream NameData;
    int i = 0;
    double beta = 2;
    NameData << "../2D_U1_Ns" << LV::Nx << "_Nt" << LV::Nt
                << "_b" << format(beta)
                << "_m" << format(mass::m0)
                << "_" << i << ".ctxt";
    

    read_binary(NameData.str(),U);

    //--------Print every rank--------//
    
    for(int i = 0; i < mpi::size; i++) {
        MPI_Barrier(mpi::cart_comm);
        if (i == mpi::rank2d) {
            std::cout << "-----------rank " << mpi::rank2d << "---------------" << std::endl;
            for(int x = 1; x<=mpi::width_x; x++){
                for(int t = 1; t<=mpi::width_t; t++){
                    int n = x*(mpi::width_t+2) + t;
                    for (int mu = 0; mu < 2; mu++) {
                        std::cout << "x " << x << " t " << t << " mu " << U.val[2*n+mu] << std::endl;
                    }
                }
                //std::cout << " " <<std::endl;
            } 
        }
    }
        
}

