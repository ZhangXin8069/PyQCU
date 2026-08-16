#ifndef AMG_H
#define AMG_H

#include "level.h"

class AlgebraicMG{
    /*
	GaugeConf GConf: Gauge configuration
    m0: Mass parameter for the Dirac matrix
    nu1: Number of pre-smoothing iterations
    nu2: Number of post-smoothing iterations
	*/
public:

	//FGMRES for the k-cycle
    class FGMRES_k_cycle : public FGMRES {
	public:
    	FGMRES_k_cycle(const int& m, const int& restarts, const double& tol, AlgebraicMG* parent, int l) : 
        parent(parent), l(l),
		FGMRES(parent->levels[l]->Ntot, 
            parent->levels[l]->DOF, 
            (parent->levels[l]->Nx+2)*(parent->levels[l]->Nt+2)*parent->levels[l]->DOF,
            1,1,
            parent->levels[l]->Nx,parent->levels[l]->Nt,
            m, restarts, tol) {}
        //FGMRES(parent->Ntot, parent->DOF, (parent->Nx+2)*(parent->Nt+2)*parent->DOF,1,1,parent->Nx,parent->Nt, 
        //AMGV::gmres_restart_length_coarse_level ,AMGV::gmres_restarts_coarse_level, AMGV::gmres_tol_coarse_level)
    	~FGMRES_k_cycle() { };
    
	private:
		AlgebraicMG* parent; //Pointer to the enclosing AMG instance
		int l; //Level
        const int Nt  = parent->levels[l]->Nt;
        const int Nx  = parent->levels[l]->Nx;
        const int DOF = parent->levels[l]->DOF;

    	//Implementation of the function that computes the matrix-vector product for the current level
    	
    	void func(const spinor& in, spinor& out) override {
        	parent->levels[l]->D_operator(in,out);
    	}
		//Preconditioning with the k-cycle
		void preconditioner(const spinor& in, spinor& out) override {
            parent->k_cycle(l,in,out); 
		}

        c_double dot(c_double* A, c_double* B) override {
            c_double local_z = 0;
            //reduction over all lattice points and spin components
            int index;
            for(int x = 1; x<=Nx; x++){
                for(int t = 1; t<=Nt; t++){
                    for(int mu=0; mu<DOF; mu++){
                    index = (x*(Nt+2) + t)*DOF + mu;
                    local_z += A[index] * std::conj(B[index]);
                    localFLOPS += ca+cm;
                    }
                }
            }
            c_double z;
            MPI_Allreduce(&local_z, &z, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, parent->levels[l]->ranks_comm);
            return z;
        }
        // out = X + lambda * Y
        void axpy(const spinor& X, const spinor& Y, const c_double &lambda,  spinor& out) override {
            int index;
            for(int x = 1; x<=Nx; x++){
                for(int t = 1; t<=Nt; t++){
                    for(int dof=0; dof<DOF; dof++){
                        index = x * (Nt+2) + t;
                        out.val[DOF*index+dof] = X.val[DOF*index+dof] + lambda * Y.val[DOF*index+dof];
                        localFLOPS += ca+cm;
                    }
                }
            }
        
        }
        // Y = lambda * X
        void scal(const c_double& lambda, const spinor& X, spinor& Y) override {
            // Y = lambda X
            int index;
            for(int x = 1; x<=Nx; x++){
                for(int t = 1; t<=Nt; t++){
                    for(int dof=0; dof<DOF; dof++){
                        index = x * (Nt+2) + t;
                        Y.val[DOF*index+dof] = lambda * X.val[DOF*index+dof];
                        localFLOPS += cm;
                    }
                }
            }
        }


	};

    AlgebraicMG(const spinor & U, const double& m0, const int& nu1, const int& nu2) 
	: U(U), m0(m0), nu1(nu1), nu2(nu2){

    	for(int l = 0; l<LevelV::levels; l++){
        	Level* level = new Level(l,U);
        	levels.push_back(level);
			//We don't need this FGMRES for the coarsest level and the finest level
			FGMRES_k_cycle* fgmres = new FGMRES_k_cycle(
                AMGV::fgmres_k_cycle_restart_length, 
                AMGV::fgmres_k_cycle_restarts, 
                AMGV::fgmres_k_cycle_tol, 
                this, 
                l);
			fgmres_k_cycle_l.push_back(fgmres);
            
    	}

    
    }    
    	
    ~AlgebraicMG() {
        for (auto ptr : levels) delete ptr;
        for (auto ptr : fgmres_k_cycle_l) delete ptr;
    }

    //Pages 84 and 85 of Rottmann's thesis explain how to implement this ...
    void setUpPhase(const int& Nit);
    //Checks orthonormalization and verifies that P^+ D P = Dc
    void testSetUp();
    //Checks that SAP is working properly
    void testSAP();
    
    // psi_l = V_cycle(l,eta_l)
    void v_cycle(const int& l, const spinor& eta_l, spinor& psi_l);

	// psi_l = K_cycle(l,eta_l)
	void k_cycle(const int& l, const spinor& eta_l, spinor& psi_l);
    //Calls K or V-cycle depending on the value of AMGV::cycle. Stand-alone solver
    void applyMultilevel(const int& it, const spinor&rhs, spinor& out,const double tol,const bool print_message);
private:    
    const spinor U;
	double m0; 
	const int nu1; const int nu2; 
    std::vector<Level*> levels; //If I try to use a vector of objects I will run out of memory
	std::vector<FGMRES_k_cycle*> fgmres_k_cycle_l; //Flexible GMRES used for the k-cycle on every level

};


//    FGMRES with a multilevel method as preconditioner

class FGMRES_AMG : public FGMRES {

public:
    FGMRES_AMG(const spinor& U, const int& m, const int& restarts, const double& tol,
        const int nu1, const int nu2,  const double& m0) : FGMRES(LV::Ntot,LV::dof,mpi::maxSizeH,
                                                            1, 1, mpi::width_x, mpi::width_t,
                                                            m, restarts, tol), 
    U(U), m0(m0), nu1(nu1), nu2(nu2), amg(U, m0, nu1, nu2) {
    //--------Set up phase for AMG---------//
    double start = MPI_Wtime();
    amg.setUpPhase(AMGV::Nit);
    double end = MPI_Wtime();
    if (mpi::rank2d == 0)
        std::cout << "[rank " << mpi::rank2d << "] Elapsed time for Set-up phase = " << end-start << " seconds" << std::endl;   
    //---------------------------//
    //Tests
    //amg.testSetUp(); //Checks that test vectors are orthonormal and that P^dagg D P = D_c at every level
    //amg.testSAP(); //Checks that SAP is working properly for every level. This compares the solution with GMRES.
    }

    ~FGMRES_AMG() { };
    
    AlgebraicMG amg; //AMG instance for the two-grid method
private:
    const spinor& U; //Gauge configuration
    const double& m0; //reference to mass parameter
    const int nu1;
    const int nu2;
    
        
    void func(const spinor& in, spinor& out) override {
        D_phi(U, in, out, m0); 
    }

    virtual void preconditioner(const spinor& in, spinor& out) = 0; 

     /*
    dot product
    */
    c_double dot(c_double* A, c_double* B) override {
        c_double local_z = 0;
        //reduction over all lattice points and spin components
        int index;
        for(int x = 1; x<=mpi::width_x; x++){
        for(int t = 1; t<=mpi::width_t; t++){
        for(int mu=0; mu<LV::dof; mu++){
            index = idx(x,t,mu);
            local_z += A[index] * std::conj(B[index]);
            localFLOPS += ca+cm;
        }
        }
        }
        c_double z;
        MPI_Allreduce(&local_z, &z, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, mpi::cart_comm);
        return z;
    }


    /*
    Complex vector addition
    */
    void axpy(const spinor& X, const spinor& Y, const c_double &lambda,  spinor& out) override {
        int index;
        for(int x = 1; x<=mpi::width_x; x++){
        for(int t = 1; t<=mpi::width_t; t++){
        for(int mu=0; mu<LV::dof; mu++){
            index = idx(x,t,mu);
            out.val[index] = X.val[index] + lambda * Y.val[index];
            localFLOPS += ca+cm;
        }
        }
        }
    }

    /*
    Scalar times a complex vector
    */
    void scal(const c_double& lambda, const spinor& X, spinor& Y) override {
    // Y = lambda X
        int index;
        for(int x = 1; x<=mpi::width_x; x++){
        for(int t = 1; t<=mpi::width_t; t++){
        for(int mu=0; mu<LV::dof; mu++){
            index = idx(x,t,mu);
            Y.val[index] = lambda * X.val[index];
            localFLOPS += ca+cm;
        }
        }
        }
    }
};


class FGMRES_AMG_k_cycle : public FGMRES_AMG {
public:
    FGMRES_AMG_k_cycle(const spinor& U, const int& m, const int& restarts, const double& tol,
        const int nu1, const int nu2, const double& m0) : FGMRES_AMG(U, m, restarts, tol,
        nu1, nu2, m0){

        }
    ~FGMRES_AMG_k_cycle() { };
            
    void preconditioner(const spinor& in, spinor& out) override {
        for(int i = 0; i<mpi::maxSizeH; i++)
            out.val[i] = 0;
		amg.k_cycle(0,in, out);
    }
};

class FGMRES_AMG_v_cycle : public FGMRES_AMG {
public:
    FGMRES_AMG_v_cycle(const spinor& U, const int& m, const int& restarts, const double& tol,
        const int nu1, const int nu2, const double& m0) : FGMRES_AMG(U, m, restarts, tol,
        nu1, nu2, m0){

        }
    ~FGMRES_AMG_v_cycle() { };
            
    void preconditioner(const spinor& in, spinor& out) override {
        for(int i = 0; i<mpi::maxSizeH; i++)
            out.val[i] = 0;
		amg.v_cycle(0,in, out);
    }
};

#endif