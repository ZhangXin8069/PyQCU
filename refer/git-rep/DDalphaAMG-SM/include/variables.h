#ifndef VARIABLES_H_INCLUDED
#define VARIABLES_H_INCLUDED
#include "config.h"
#include <iostream>
#include <vector>
#include <complex>
#include <iomanip>
#include "mpi.h"

typedef std::complex<double> c_double;
extern double pi;
extern c_double I_number; //imaginary number

namespace mass{extern double m0;}

//FLOPs involved in different complex double operations
constexpr long long int ca = 2;     //  + complex addition = 2 FLOPs
constexpr long long int cm = 6;     //  * complex multiplication = 6 FLOPs
constexpr long long int cd = 11;    //  / complex division = 11 FLOPs (6 mul + 3 add + 2 div)
constexpr long long int da = 1;     //  double + double = 1 FLOP
constexpr long long int dm = 1;     //  double * double = 1 FLOP
constexpr long long int dd = 1;     //  double / double = 1 FLOP
constexpr long long int dcm = 2;    //  double * complex = 2 FLOPs
constexpr long long int dcd = 7;    //  double / complex_double = 7 FLOPs (4 mul + 1 add + 2 div)
constexpr long long int dsq = 1;    //  sqrt(double) = 1 FLOP

extern long long int FLOPS;
extern long long int localFLOPS;

namespace iter_counters{
    extern long long int kCycleIt;
    extern long long int vCycleIt;
    extern long long int CGIt;
    extern long long int BiCGIt;
    extern long long int GMRESIt;
    extern long long int SAPIt;
    extern long long int FGMRES_SAPIt;
}
namespace flop_counters{
    extern long long int kCycleFlops;
    extern long long int vCycleFlops;
    extern long long int kCycleSetUpFlops;
    extern long long int vCycleSetUpFlops;
    extern long long int CGFlops;
    extern long long int BiCGFLops;
    extern long long int GMRESFlops;
    extern long long int SAPFlops;
    extern long long int FGMRES_SAPFlops;

}


//Datatypes for reading/writing gauge confs and rhs
extern MPI_Datatype global_conf_type;
extern MPI_Datatype global_conf_resized;
extern MPI_Datatype local_conf_type;
extern MPI_Datatype local_conf_resized;

//------------mpi settings-----------------//
namespace mpi{
    extern int rank;
    extern int size; 
    extern int maxSize;
    extern int maxSizeH; //maxSize with halos included
    extern int sitesH;
    extern int ranks_x;
    extern int ranks_t;
    extern int width_x;
    extern int width_t;
    extern int rank2d; //Rank id in the 2D communicator
    extern int coords[2];
    extern int top; 
    extern int bot; 
    extern int right; 
    extern int left;
    //Diagonal ranks necessary for staples
    extern int bot_left;
    extern int bot_right;
    extern int top_left;
    extern int top_right;
    extern MPI_Comm cart_comm;
    

    //Hardcoding the coarse levels for 4 ranks ...
    constexpr int ranks_coarse_level = 4;               //Number of working ranks on the coarse levels
    constexpr int coarse_ranks_x     = 2;               //Number of working ranks on the coarse level x-dir
    constexpr int coarse_ranks_t     = 2;               //Number of working ranks on the coarse level t-dir
    extern MPI_Group cart_comm_group;                   //Group corresponding to cart_comm.
    extern MPI_Group* coarse_group;                     //Group of ranks associated with rank_c on the coarse grid.
    extern MPI_Comm* coarse_comm;                       //New communicator for agglomerated ranks
    extern MPI_Comm comm_coarse_level;                  //Communicator for the working ranks on the coarse level
    extern int top_c;                                     
    extern int bot_c; 
    extern int right_c; 
    extern int left_c; 
    extern int coarse_rank2d;                           //Rank id in comm_coarse_level
    extern int ranks_x_c;                               //Number of x agglomerated ranks
    extern int ranks_t_c;                               //Number of t agglomerated ranks
    extern int size_c;                                  //ranks_x_c * ranks_t_c

    extern int* rank_dictionary;  //Coarse communicator corresponding to rank2d

    extern MPI_Datatype* column_type;
}


//------------Lattice parameters--------------//
namespace LV {
    //Lattice dimensions//
    constexpr int Nx= NS; //We extract this value from config.h
    constexpr int Nt = NT; //We extract this value from config.h
    constexpr int Ntot = Nx*Nt; //Total number of lattice points
    constexpr int dof = 2;
}

//-----------CG solver------------//
namespace CG{
    extern int max_iter; //Maximum number of iterations for the conjugate gradient method
    extern double tol; //Tolerance for convergence
}

//------------------BiCGstab--------------------//
namespace BiCG{
    extern int max_iter; //Maximum number of iterations for the conjugate gradient method
    extern double tol; //Tolerance for convergence
}

//------------Schwarz alternating procedure parameters--------------//
namespace SAPV {
    using namespace LV; 
    constexpr int sap_block_x = 4; //Default values for SAP as a preconditioner for FGMRES (not multigrid)
    constexpr int sap_block_t = 4; 
    //Parameters for GMRES in SAP
    extern int sap_gmres_restart_length; //GMRES restart length for the Schwarz blocks.
    extern int sap_gmres_restarts; //GMRES iterations for the Schwarz blocks
    extern double sap_gmres_tolerance; //GMRES tolerance for the Schwarz blocks 
    extern double sap_tolerance; //tolerance for the SAP method
}

//------------Parameters for AMG--------------//
namespace AMGV{
    extern int SAP_test_vectors_iterations; //Number of SAP iterations to smooth test vectors
    //Parameters for the coarse level solver
    extern int gmres_restarts_coarse_level; //restart length for GMRES at the coarse level
    extern int gmres_restart_length_coarse_level; //GMRES restart length for the coarse level
    extern double gmres_tol_coarse_level; //GMRES tolerance for the coarse level
    //Parameters for GMRES as a smoother (the default AMG version uses SAP)
    extern int gmres_restarts_smoother; //GMRES iterations for the smoother

    extern int nu1; //Pre-smoothing iterations
    extern int nu2; //Post-smoothing iterations
    extern int Nit; //Number of iterations for improving the interpolator 

    //Outer fgmres solver
    extern int fgmres_k_cycle_restart_length;
    extern int fgmres_k_cycle_restarts;
    extern double fgmres_k_cycle_tol;
    extern int cycle; //Cycling stratey. Cycle = 0 -> V-cycle, = 1 --> K-cycle
}

//--------------Parameters for outer FGMRES--------------//
namespace FGMRESV {
    extern double fgmres_tolerance;     //Tolerance for FGMRES
    extern int fgmres_restart_length;   //Restart length for FGMRES
    extern int fgmres_restarts;         //Number of restarts for FGMRES
}

namespace LevelV{
    //Description on variables.cpp
    extern int levels;
    extern int maxLevel;
    extern int* BlocksX;
    extern int* BlocksT;
    extern int* Ntest; 
    extern int* Nagg; 
    extern int* NBlocks;
    extern int* Nsites;
    extern int* NxSites;
    extern int* NtSites;
    extern int* NxSitesH;
    extern int* NtSitesH;
    extern int* NSitesH;
    extern int* DOF; 
    extern int* Colors; 
    extern int* SAP_Block_x; 
    extern int* SAP_Block_t; 
    extern int* SAP_elements_x;  
    extern int* SAP_elements_t;
    extern int* SAP_variables_per_block; 
    extern int* GMRES_restart_len;
    extern int* GMRES_restarts;
    extern double* GMRES_tol;
    extern int* RanksX;
    extern int* RanksT;
    extern MPI_Comm* D_operator_communicator; 
}



//Flattened spinor
struct spinor {
    c_double* val;  //Array with values
    int size;       //Size of array 
    //Constructor
    spinor(int N = LV::Ntot) : size(N) {
        val = new c_double[N]();
    }

    //Copy constructor (deep copy)
    spinor(const spinor& other) : size(other.size) {
        val = new c_double[size];
        std::copy(other.val, other.val + size, val);
    }

    //Assignment operator (deep copy)
    spinor& operator=(const spinor& other) {
        if (this != &other) {
            if (size != other.size) {
                delete[] val;
                size = other.size;
                val = new c_double[size];
            }
            std::copy(other.val, other.val + size, val);
        }
        return *this;
    }

    // Destructor
    ~spinor() {
        delete[] val;
    }

    inline void clearBuffer(){
        for(int n = 0; n<size; n++){
            val[n] = 0;
        }
    }
};


//Boundary conditions with padding.
extern int* lpb;
extern int* rpb;
extern c_double* lsign;
extern c_double* rsign;

//Allocation of many arrays used throughout the code 
void allocate_lattice_arrays();
void free_lattice_arrays();




#endif 