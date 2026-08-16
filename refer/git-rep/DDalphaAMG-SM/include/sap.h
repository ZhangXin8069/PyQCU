#ifndef SAP_H
#define SAP_H
#include "fgmres.h"


/*
    SAP class 
    The local matrix-vector operation and the global matrix-vector operation have to be defined
    in a subclass.
*/
class SAP_C {
public: 
    //--------- Nested class for GMRES_D_B operator ---------//
    class GMRES_D_B : public FGMRES {
        public:GMRES_D_B(
                const int Ntot, const int dofs, const int maxSizeH,
                const int x_ini, const int t_ini, 
                const int x_fin, const int t_fin,
                const int m, const int restarts, const double tol,SAP_C* parent) :
        FGMRES( Ntot,dofs,maxSizeH,
                x_ini, t_ini, x_fin, t_fin,
                m, restarts, tol), parent(parent), dofs(dofs), xfin(x_fin), tfin(t_fin) {
        };
        ~GMRES_D_B() { };

        private:
        SAP_C* parent; //Pointer to the parent SAP_C object

        int xfin, tfin, dofs; 
                
        void func(const spinor& in, spinor& out) override { 
            parent->funcLocal(in, out);
        }
        void preconditioner(const spinor& in, spinor& out) override { 
            out = std::move(in); //No preconditioning
        }
        
        //Dot product inside the blocks
        c_double dot(c_double* A, c_double* B) {
            c_double z = 0.0;
            int index;
            for(int x = 1; x<=xfin; x++){
                for(int t = 1; t<=tfin; t++){
                    for(int dof=0; dof<dofs; dof++){
                        index = x * (tfin+2) + t;
                        z += A[dofs*index+dof] * std::conj(B[dofs*index+dof]);
                        localFLOPS += ca+cm;
                    }
                }
            }
            return z;
        }

        //Axpy inside the blocks
        void axpy(const spinor& X, const spinor& Y, const c_double &lambda,  spinor& out) {
            int index;
            for(int x = 1; x<=xfin; x++){
                for(int t = 1; t<=tfin; t++){
                    for(int dof=0; dof<dofs; dof++){
                        index = x * (tfin+2) + t;
                        out.val[dofs*index+dof] = X.val[dofs*index+dof] + lambda * Y.val[dofs*index+dof];
                        localFLOPS += ca+cm;
                    }
                }
            }
        }
    
        //Scal inside the blocks
        void scal(const c_double& lambda, const spinor& X, spinor& Y) {
            // Y = lambda X
            int index;
            for(int x = 1; x<=xfin; x++){
                for(int t = 1; t<=tfin; t++){
                    for(int dof=0; dof<dofs; dof++){
                        index = x * (tfin+2) + t;
                        Y.val[dofs*index+dof] = lambda * X.val[dofs*index+dof];
                        localFLOPS += cm;
                    }
                }
            }
        }

    };
    
    GMRES_D_B gmres_DB;
    //--------------------------------------------------------//

    //------------------------------------//
    //Constructor
    SAP_C(const int Nx, const int Nt,const int block_x,const int block_t, const int spins,
        const int colors) :
    Nt(Nt), Nx(Nx), Block_x(block_x), Block_t(block_t), spins(spins), colors(colors),
    gmres_DB(Nx/block_x * Nt/block_t, spins*colors, (Nx/block_x+2)*(Nt/block_t+2)*(spins*colors),
                1, 1, 
                Nx/block_x, Nt/block_t,
                SAPV::sap_gmres_restart_length, SAPV::sap_gmres_restarts, SAPV::sap_gmres_tolerance,this)  
    {
        
        x_elements = Nx/block_x, t_elements = Nt/block_t;                   //Number of elements in the x and t direction
        NBlocks                 =   Block_x * Block_t;                      //Number of Schwarz blocks
        dofs                    =   spins*colors;                           //Degrees of freedom per lattice site
        Ntot_original_w_halo    =   (Nx+2)*(Nt+2);
        Ntot_no_halo            =   x_elements*t_elements;                  //Number of lattice sites inside block without halo
        Ntot_w_halo             =   (x_elements+2) * (t_elements+2);        //""  ""  with halos
        Nvars_no_halo           =   Ntot_no_halo * dofs;                    //Number of variables inside block without halo
        Nvars_w_halo            =   Ntot_w_halo * dofs;                     //""   "" with halos
        coloring_blocks         =   (NBlocks != 1) ? NBlocks/2: 1;          //Number of red or black blocks


        /*if (mpi::rank2d == 0){
            std::cout << "SAP_C initialized with " << NBlocks << " blocks, each with " << Ntot_no_halo << " lattice points." << std::endl;
            std::cout << "x_elements: " << x_elements << ", t_elements: " << t_elements << std::endl;
            std::cout << "variables_per_block: " << Nvars_no_halo << ", coloring_blocks: " << coloring_blocks << std::endl;
        }*/

        //Lattice coordinate of elements in each block
        Blocks      = std::vector<std::vector<int>>(block_x*block_t, std::vector<int>((x_elements+2)*(t_elements+2), 0));
        
        //RedBlocks (BlackBlocks) is a vector with the indices of the red blocks (black blocks)
        RedBlocks   = std::vector<int>(coloring_blocks, 0);     //Red blocks
        BlackBlocks = std::vector<int>(coloring_blocks, 0);     //Black blocks       
        SchwarzBlocks(); //Initialize the Schwarz blocks
     };

    /*
        Solves D x = v using SAP .
        v: right-hand side,
        x: output --> The initial given value is considered the initial guess.
        nu: number of iterations,

        The convergence criterion is ||r|| < ||phi|| * tol
    */
    int SAP(const spinor& v,spinor &x,const int& nu, const double&tol, const bool print);

    int nu;
    int Nt, Nx;                     //Dimensions of the original lattice without halos
    int dofs;                       //Dofs
    double tol;                     //SAP convergence tolerance
    int Block_x, Block_t;           //Block dimensions for the SAP method
    int x_elements, t_elements;     //Number of elements in the x and t direction
    int NBlocks, coloring_blocks;   //Total number of blocks, total number of Red (or Black) blocks
    int colors, spins;              //Number of colors and spins
    int Ntot_no_halo, Ntot_w_halo, Nvars_no_halo, Nvars_w_halo, Ntot_original_w_halo; //Check inside the constructor for definition.
    int current_block;

    std::vector<std::vector<int>> Blocks; //SAP_Blocks[number_of_block][vectorized_coordinate of the lattice point]
    //The vectorization does not take into account the spin index, since both spin indices are in the same block.
    std::vector<int> RedBlocks;     //Block index for the red blocks
    std::vector<int> BlackBlocks;   //Block index for the black blocks


private: 
    /*
        Build the Schwarz blocks
        Function only has to be called once before using the SAP method.
    */
    void SchwarzBlocks();

    /*
        A_B v = I_B * D_B^-1 * I_B^T v --> Extrapolation of D_B^-1 to the original lattice.
        dim(v) = 2 * Ntot, dim(x) = 2 Ntot
        v: input, x: output 
    */
    void I_D_B_1_It(const spinor& v, spinor& x, const int& block);

    /*
        Matrix-vector global operation
        This is defined in the derived classes
    */
    virtual void funcGlobal(spinor& in, spinor& out) = 0; 

    /*
        Matrix-vector operation restricted to the blocks
    */
    virtual void funcLocal(const spinor& in, spinor& out) = 0; 

    /*
    Dot product
    */
    virtual c_double dot(const spinor& X, const spinor& Y) = 0;

};


//SAP for the finest level
class SAP_fine_level : public SAP_C {
public:
    SAP_fine_level(const int Nx, const int Nt,const int block_x,const int block_t, const int spins, const int colors) :
    SAP_C(Nx, Nt,block_x,block_t, spins, colors) {
    }

    void set_params(const spinor& conf, const double& bare_mass){
        U = &conf;
        m0 = bare_mass;
    }

private: 
    const spinor* U; 
    double m0; 


    void D_B(const spinor& U, const spinor& v, spinor& x, const double& m0,const int& block);

    void funcGlobal(spinor& in, spinor& out) override { 
        D_phi(*U, in, out,m0);
    }

    void funcLocal(const spinor& in, spinor& out) override { 
        D_B(*U, in, out, m0,this->current_block);
    }

    c_double dot(const spinor& X, const spinor& Y) override {
        c_double local_z = 0;
        //reduction over all lattice points and spin components
        int index;
        for(int x = 1; x<=mpi::width_x; x++){
            for(int t = 1; t<=mpi::width_t; t++){
                for(int mu=0; mu<LV::dof; mu++){
                    index = idx(x,t,mu);
                    local_z += X.val[index] * std::conj(Y.val[index]);
                    localFLOPS += ca+cm;
                }
            }
        }
        c_double z;
        MPI_Allreduce(&local_z, &z, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, mpi::cart_comm);
        return z;
    }

};


/*
    FGMRES for the fine level
*/
class FGMRES_SAP : public FGMRES {
    public:
    FGMRES_SAP(  const int Ntot, const int dofs, const int maxSizeH,
                        const int x_ini, const int t_ini, 
                        const int x_fin, const int t_fin,
                        const int m, const int restarts, const double tol,
                        const spinor& U, const double& m0) : FGMRES(Ntot,dofs,maxSizeH,
                        x_ini, t_ini, x_fin, t_fin,
                        m, restarts, tol), U(U), m0(m0), Nx(x_fin-x_ini+1), Nt(t_fin-t_ini+1), nu(1), message(false),
                        sap_preconditioner(Nx,  Nt, SAPV::sap_block_x, SAPV::sap_block_t, 2, 1)
                        {  sap_preconditioner.set_params(U, m0); };
    ~FGMRES_SAP() { };
    
private:
    const spinor& U; //reference to Gauge configuration. This is to avoid copying the matrix
    const double m0; //reference to mass parameter'
    const bool message;
    const int nu;
    const int Nx;
    const int Nt;

    SAP_fine_level sap_preconditioner;

    /*
    Implementation of the function that computes the matrix-vector product for the fine level
    */
    void func(const spinor& in, spinor& out) override {
        D_phi(U, in, out, m0);
    }

    void preconditioner(const spinor& in, spinor& out) override {
        //SAP preconditioner
        for(int x = 1; x<=Nx; x++){
        for(int t = 1; t<=Nt; t++){
        for(int mu=0; mu<LV::dof; mu++){
            out.val[idx(x,t,mu)] = 0;
        }
        }
        }
        sap_preconditioner.SAP(in, out, nu, SAPV::sap_tolerance, message);
    }

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
            localFLOPS += cm;
        }
        }
        }
    }

};

        
#endif