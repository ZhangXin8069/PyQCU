#ifndef LEVEL_H
#define LEVEL_H

#include <algorithm>
#include "sap.h"

/*
    One level of the AMG method
*/
class Level {
public:   
    //SAP for smoothing D_operator. 
    //-------------------------------Nested class-------------------------------//
    class SAP_level_l : public SAP_C {
    public:
        SAP_level_l(Level* parent) :
        parent(parent), 
        SAP_C(parent->Nx, parent->Nt,               //Dimensions on the current rank
            LevelV::SAP_Block_x[parent->level],     //SAP blocks on the x direction
            LevelV::SAP_Block_t[parent->level],     //SAP blocks on the t direction
            2, parent->colors) 
        {        
            /*
            if (mpi::rank2d == 0){
            std::cout << "SAP in level " << parent->level << " initialized" << std::endl;
            std::cout << "SAP Block_x  " << LevelV::SAP_Block_x[parent->level] << "    SAP Block_t " 
            << LevelV::SAP_Block_t[parent->level] << std::endl;
            std::cout << "Nx " << parent->Nx << "   Nt  " << parent->Nt << std::endl;
            }
            */
        }
    private: 
        Level* parent; //Parent class

        //Global D operation
        void funcGlobal(spinor& in, spinor& out) override { 
            parent->D_operator(in, out); //Dirac operator at the current level
        }

        //Local D operations
        void D_local(const spinor& in, spinor& out, const int& block);

        void funcLocal(const spinor& in, spinor& out) override { 
            D_local( in, out,this->current_block);
        }

        c_double dot(const spinor& X, const spinor& Y) override{
             c_double local_z = 0;
            //reduction over all lattice points and spin components
            int index;
            for(int x = 1; x<=parent->Nx; x++){
                for(int t = 1; t<=parent->Nt; t++){
                    for(int mu=0; mu<parent->DOF; mu++){
                        index = (x*(parent->Nt+2) + t)*parent->DOF + mu;
                        local_z += X.val[index] * std::conj(Y.val[index]);
                        localFLOPS += ca+cm;
                    }
                }
            }
            c_double z;
            MPI_Allreduce(&local_z, &z, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, parent->ranks_comm);
            return z;
        }

    };
    SAP_level_l* sap_l; 
    //----------------------------------------------------------------------------//
    //GMRES for the current level. We use it for solving the coarsest system. We could use it for smoothing as well.
    
    class GMRES_level_l : public FGMRES {
	public:
    	GMRES_level_l(Level* parent) : parent(parent), 
		FGMRES(parent->Ntot, parent->DOF, (parent->Nx+2)*(parent->Nt+2)*parent->DOF,1,1,parent->Nx,parent->Nt, 
        AMGV::gmres_restart_length_coarse_level ,AMGV::gmres_restarts_coarse_level, AMGV::gmres_tol_coarse_level) {}
    
    	~GMRES_level_l() { };
    
	private:
		Level* parent; //Pointer to the enclosing AMG instance
        const int Nt  = parent->Nt;
        const int Nx  = parent->Nx;
        const int DOF = parent->DOF;
    	
    	//Implementation of the function that computes the matrix-vector product for the fine level
    	
    	void func(const spinor& in, spinor& out) override {
        	parent->D_operator(in,out);
    	}
		//No preconditioning for the coarsest level
		void preconditioner(const spinor& in, spinor& out) override {
            out = std::move(in); //Identity operation
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
            MPI_Allreduce(&local_z, &z, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, parent->ranks_comm);
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

	GMRES_level_l* gmres_l;
    
    //----------------------------------------------------------------------------//
    
    //Level Constructor
    Level(const int& level, const spinor& U) : level(level), U(U),
        Nx(LevelV::NxSites[level]/LevelV::RanksX[level]),
        Nt(LevelV::NtSites[level]/LevelV::RanksT[level]),
        Ntot(Nx*Nt)
    {
        //Gauge links to define D_operator (matrix problem at this level). We define them with halos.
        int Ntot_halo = (Nx+2)*(Nt+2);
        Nx_coarse_rank = Nx;
        Nt_coarse_rank = Nt;
        xblocks_per_coarse_rank = xblocks_per_rank; //Number of lattice blocks inside one MPI rank on the coarse grid.
	    tblocks_per_coarse_rank = tblocks_per_rank;
        blocks_per_coarse_rank  = blocks_per_rank;
        ranks_comm = LevelV::D_operator_communicator[level];

        //rankID in ranks_comm  
        ranks_on_t = mpi::ranks_t;
        if (ranks_comm != MPI_COMM_NULL){
            int result;
            MPI_Comm_compare(ranks_comm, mpi::cart_comm, &result);
    	    if (result != MPI_IDENT) 
                ranks_on_t = mpi::coarse_ranks_t;
            MPI_Comm_rank(ranks_comm, &rank_in_comm);
        }
        
        //Test vectors
        tvec        = std::vector<spinor>(Ntest,spinor(Ntot_halo*DOF));
        tvec_copy   = std::vector<spinor>(Ntest,spinor(Ntot_halo*DOF));

        //Coarse gauge links
        G1 = spinor(Ntot_halo*2*2*colors*colors);
        G2 = spinor(Ntot_halo*2*2*colors*colors*2);
        G3 = spinor(Ntot_halo*2*2*colors*colors*2);
        //Arrays needed for gather/scatter data for rank coarsening
        counts_spinor   = new int[mpi::size_c];  
        displs_spinor   = new int[mpi::size_c];
        counts_G1       = new int[mpi::size_c];  
        displs_G1       = new int[mpi::size_c];
        counts_G2G3     = new int[mpi::size_c];  
        displs_G2G3     = new int[mpi::size_c];

        //In case we have to agglomerate the ranks
        if (ranks_per_block>1){
            Nt_coarse_rank   = Nt*mpi::ranks_t_c; //Nt sites on the coarse rank 
            Nx_coarse_rank   = Nx*mpi::ranks_x_c; 
            xblocks_per_coarse_rank = LevelV::BlocksX[level] / mpi::coarse_ranks_x; //Number of lattice blocks inside one MPI rank on the coarse grid.
	        tblocks_per_coarse_rank = LevelV::BlocksT[level] / mpi::coarse_ranks_t;
            blocks_per_coarse_rank  = xblocks_per_coarse_rank * tblocks_per_coarse_rank;   
            x_elements = Nx_coarse_rank/xblocks_per_coarse_rank;
            t_elements = Nt_coarse_rank/tblocks_per_coarse_rank;         
            
            //Buffers for rank coarsening
            gathered_tvec = std::vector<spinor>(Ntest,spinor((Nt_coarse_rank+2)*(Nx_coarse_rank+2)*DOF));//Gather test vectors data from other ranks
		    gathered_out  = spinor((Nt_coarse_rank+2)*(Nx_coarse_rank+2)*DOF);
            gathered_v    = spinor((Nt_coarse_rank+2)*(Nx_coarse_rank+2)*DOF);
            gathered_G1   = spinor((Nt_coarse_rank+2)*(Nx_coarse_rank+2)*DOF*DOF);
            gathered_G2   = spinor((Nt_coarse_rank+2)*(Nx_coarse_rank+2)*2*DOF*DOF);
            gathered_G3   = spinor((Nt_coarse_rank+2)*(Nx_coarse_rank+2)*2*DOF*DOF);

            //Halo exchange for rank coarsening
            MPI_Type_vector(Nx_coarse_rank, DOF, (Nt_coarse_rank+2)*DOF, MPI_DOUBLE_COMPLEX, &coarse_column_type);
            MPI_Type_commit(&coarse_column_type); 
            makeDatatypes();
        }
        
        if (level == 0)
            makeDirac(); //Initialize Dirac Operator on the fine grid
        else
            defineColumnType(level, Nx, Nt,DOF); //DataType needed for the halo exchange at level l for D_operator

        //For solving the coarsest level
        if (level == LevelV::maxLevel)
            gmres_l = new GMRES_level_l(this);
        
        /*
        if (mpi::rank2d == 0){
            std::cout << "level " << level << " colors " << colors << "  test vectors " << Ntest  << std::endl;
            std::cout << "ranks per block        " << ranks_per_block << std::endl;
            std::cout << "blocks_per_coarse_rank " << blocks_per_coarse_rank << std::endl;
            std::cout << "Nt coarse rank         " << Nt_coarse_rank << std::endl;
            std::cout << "Nx coarse rank         " << Nx_coarse_rank << std::endl;
            std::cout << "tblocks_per_coarse_rank       " << tblocks_per_coarse_rank << std::endl;
            std::cout << "xblocks_per_coarse_rank       " << xblocks_per_coarse_rank << std::endl;
        }
        */

        sap_l = new SAP_level_l(this);
    };

    ~Level() {
        delete[] counts_spinor;  
        delete[] displs_spinor;
        delete[] counts_G1;  
        delete[] displs_G1;
        delete[] counts_G2G3;  
        delete[] displs_G2G3;
        delete sap_l;
        if (level == LevelV::maxLevel)
            delete gmres_l;

    }

    const spinor U; //Gauge configuration
    
    //------------For gathering and scattering data when rank coarsening is necessary-----------//
    std::vector<spinor> tvec;       //[Ntest][(Nt+2).(Nx+2).colors.spins]
    std::vector<spinor> tvec_copy;  
    //Buffers used in P and P^+ implementations. 
    std::vector<spinor> gathered_tvec;	//Gather test vectors data from other ranks
	spinor gathered_out;
    spinor gathered_v;
    spinor gathered_G1;
    spinor gathered_G2;
    spinor gathered_G3;
    MPI_Datatype local_domain_spinor;
    MPI_Datatype local_domain_spinor_resized;
    MPI_Datatype coarse_domain_spinor;
    MPI_Datatype coarse_domain_spinor_resized;
    int* counts_spinor;  
    int* displs_spinor;
  
    MPI_Datatype local_domain_linkG1;
    MPI_Datatype local_domain_linkG1_resized;
    MPI_Datatype coarse_domain_linkG1;
    MPI_Datatype coarse_domain_linkG1_resized;
    int* counts_G1;  
    int* displs_G1;

    MPI_Datatype local_domain_linkG2G3;
    MPI_Datatype local_domain_linkG2G3_resized;
    MPI_Datatype coarse_domain_linkG2G3;
    MPI_Datatype coarse_domain_linkG2G3_resized;
    int* counts_G2G3;  
    int* displs_G2G3;
    //--------------------------------------------------------------------------------------------//

    MPI_Comm ranks_comm; //Communicator among the ranks on the current level 
    int rank_in_comm; //rankID on ranks_comm
    int ranks_on_t;   //Number of ranks on t in ranks_comm, necessary for l-sign and r-sign
    MPI_Datatype coarse_column_type; 

    const int level; 
    const int xblocks_per_rank  = (level != LevelV::maxLevel) ? LevelV::BlocksX[level]/LevelV::RanksX[level] : 1; //Number of blocks on x inside the current rank
    const int tblocks_per_rank  = (level != LevelV::maxLevel) ? LevelV::BlocksT[level]/LevelV::RanksT[level] : 1; //Number of blocks on t inside the current rank
    const int blocks_per_rank   = xblocks_per_rank*tblocks_per_rank; //Number of blocks inside the rank
    
    //-------For the case when the blocks cross the ranks--------//
    //Number of fine ranks inside a block
    const int xranks_per_block  = (level != LevelV::maxLevel) ? LevelV::RanksX[level]/LevelV::BlocksX[level] : 1; //x-ranks inside a block 
    const int tranks_per_block  = (level != LevelV::maxLevel) ? LevelV::RanksT[level]/LevelV::BlocksT[level] : 1; //t-ranks inside a block 
    const int ranks_per_block   = xranks_per_block*tranks_per_block;  //Number of ranks inside a lattice block
    //Number of blocks inside a coarse rank. When ranks_per_block > 1, they differ from ranks_per_block
    int xblocks_per_coarse_rank; //Number of lattice blocks inside one MPI rank on the coarse grid.
	int tblocks_per_coarse_rank;
    int blocks_per_coarse_rank;
    int Nt_coarse_rank; //Nt sites on the coarse rank 
    int Nx_coarse_rank; 
    //----------------------------------------------------------//

    const int Nx;   //Nx on the fine grid in rank r (no halo)
    const int Nt;   //Nt on the fine grid in rank r (no halo)
    const int Ntot; //Nx*Nt
    const int colors = LevelV::Colors[level];   //Number of colors at this level
    const int Ntest = (level != LevelV::maxLevel) ? LevelV::Ntest[level]: 1;     //Number of test vectors to go to the next level
    const int DOF = 2*colors;         //Degrees of freedom at each lattice site at this level

    int x_elements = (level != LevelV::maxLevel && ranks_per_block<=1) ?  Nx / xblocks_per_rank: 1;
    int t_elements = (level != LevelV::maxLevel && ranks_per_block<=1) ?  Nt / tblocks_per_rank: 1; 
    const int sites_per_block = x_elements * t_elements;
    const int NBlocks = (level != LevelV::maxLevel) ? LevelV::NBlocks[level]: 1; //Number of lattice blocks 
    
    //At level = 0 these vectors represent the gauge links.
    //At level > 1 they are the coarse gauge links generated in the previous level
    spinor G1; 
    spinor G2; 
    spinor G3; 

    void makeType(const int dofs, MPI_Datatype& local_domain, MPI_Datatype& local_domain_resized, 
        MPI_Datatype& coarse_domain, MPI_Datatype& coarse_domain_resized);
    void makeDatatypes();
    void gather_to_coarse_rank(const spinor& local_spinor, spinor& coarse_spinor, const int dofs);
    void scatter_to_local_rank_from_coarse_rank(const spinor& coarse_spinor, spinor& local_spinor, const int dofs);

/*
	Prolongation operator times a spinor x = P v
	x_i = P_ij v_j. dim(P) = DOF Nsites x Ntest Nagg, 
	dim(v) = [NBlocks][2*Ntest], dim(x) = [Nsites][DOF]
    */
    void P_vc(const spinor& vc,spinor& out);

    /*
	Restriction operator times a spinor on the coarse grid, x = P^H v
	x_i = P^dagg_ij v_j. dim(P^dagg) =  Ntest Nagg x DOF Nsites,
	dim(v) = [Nsites][DOF], dim(x) = [NBlocks][2*Ntest] 
    */
    void Pdagg_v(const spinor& v,spinor& out);

    void orthonormalize();      //Local orthonormalization of the test vectors
    void checkOrthogonality();  //Check orthogonality of the test vectors

    //Index functions for gauge links. These correspond to the current level
	//get index for A_coeff 1D array
    //[A(x)]^{alf,bet}_{c,b} --> A_coeff[x][alf][bet][c][b]
	inline int getG1index(const int& x, const int& alf, const int& bet, const int& c, const int& b){
		return x * 2 * 2 * colors * colors 
        + alf * 2 * colors * colors 
        + bet * colors * colors
        + c * colors 
        + b;
	}
	//[B_mu(x)]^{alf,bet}_{c,b}  --> B_coeff[x][alf][bet][c][b][mu]
    //[C_mu(x)]^{alf,bet}_{c,b}  --> C_coeff[x][alf][bet][c][b][mu]
	inline int getG2G3index(const int& x, const int& alf, const int& bet, const int& c, const int& b, const int& mu){
        return x * 2 * 2 * colors * colors * 2 
        + alf * 2 * colors * colors * 2 
        + bet * colors * colors * 2
        + c * colors * 2 
        + b * 2 
        + mu;
    }
    	
    //Index functions for coarse gauge links. These correspond to the next level, but are generated here (not stored)
	//get index for A_coeff 1D array
    //[A(x)]^{alf,bet}_{c,b} --> A_coeff[x][alf][bet][c][b]
	inline int getAindex(const int& block, const int& alf, const int& bet, const int& c, const int& b){
		return block * 2 * 2 * Ntest * Ntest 
        + alf * 2 * Ntest * Ntest 
        + bet * Ntest * Ntest
        + c * Ntest 
        + b;
	}
	//[B_mu(x)]^{alf,bet}_{c,b}  --> B_coeff[x][alf][bet][c][b][mu]
    //[C_mu(x)]^{alf,bet}_{c,b}  --> C_coeff[x][alf][bet][c][b][mu]
	inline int getBCindex(const int& block, const int& alf, const int& bet, const int& c, const int& b, const int& mu){
        return block * 2 * 2 * Ntest * Ntest * 2 
        + alf * 2 * Ntest * Ntest * 2 
        + bet * Ntest * Ntest * 2
        + c * Ntest * 2 
        + b * 2 
        + mu;
    }

    //Given n on the fine grid with halo, return b on the coarse grid with halo
    inline void getLatticeBlock(const int& n, int& block){
        //printf("Inside getLatticeBlock\n");
        int x = n / (Nt_coarse_rank+2); //x coordinate of the lattice point 
        int t = n % (Nt_coarse_rank+2); //t coordinate of the lattice point
       // printf("(x,t)=(%d,%d)\n",x,t);
        //Reconstructing the block 
        int block_x, block_t;
        //This assumes that x and t are not touching the halos.
        block_x = (x-1) / x_elements + 1; //Block index in the x direction
        block_t = (t-1) / t_elements + 1; //Block index in the t direction

        //If the coordinates touch the edges of the halo we need to shift the indices. 
        if ( t == 0 )
            block_t -= 1; 
        else if (x == 0)
            block_x -= 1;

       // printf("(block_x,block_t)=(%d,%d)\n",block_x,block_t);
        block = block_x * (tblocks_per_coarse_rank+2) + block_t; //Block index in the SAP method
    }

    //returns (x,t) + hat{mu} on the current level
    inline int rpb_l(const int& x, const int& t, const int& mu, const int& Nx, const int& Nt){
        int xp = x+1;
        int tp = t+1;
        if (mu == 0)
            return x*(Nt+2) + tp; //Right
        else if (mu == 1)
            return xp*(Nt+2) + t; //Down        
        else{
            std::cout << "Give a valid value of mu in function rpb_l" << std::endl;
            exit(1); 
        }
    }

    //returns (x,t)-hat{mu} on the current level
    inline int lpb_l(const int& x, const int& t, const int& mu, const int& Nx, const int& Nt){
        int xm = x-1;
        int tm = t-1;
        if (mu == 0)
            return  x*(Nt+2)+tm; //Left
        else if (mu == 1)
            return xm*(Nt+2)+t;  //Up
        else{
            std::cout << "Give a valid value of mu in function lpb_l" << std::endl;
            exit(1); 
        }
    }

    inline c_double rsign_l(const int& t, const int& mu){
        c_double sign=1;
        if ((rank_in_comm+1) % ranks_on_t == 0 && mu == 0)
			sign = (t == Nt) ? -1 : 1;     //sign for the "right" boundary in time
        return sign;
    }

    inline c_double lsign_l(const int& t, const int& mu){
        c_double sign=1;
        if (rank_in_comm % ranks_on_t == 0 && mu == 0)
			sign = (t == 1) ? -1 : 1;     //sign for the "left" boundary in time	 
        return sign;
    }
    
    //Creates G1, G2 and G3
    void makeDirac();

    //Make coarse gauge links. They will be used in the next level as G1, G2 and G3.
    void makeCoarseLinks(Level& next_level); //& A_coeff,c_vector& B_coeff, c_vector& C_coeff);

    //Exchange halo for spinor v 
    void exchange_halo_l(const spinor& v,const int& Nx, const int& Nt, const MPI_Datatype& column, const MPI_Comm& comm);

    /*
    Matrix-vector operation that defines the level l.
    For instance, at level = 0, D_operator is just the Dirac operator
    at level = 1 D_operator is Dc
    at level = 2 D_operator is (Dc)_c ...
    */
    void D_operator(const spinor& v, spinor& out);

    
};

#endif