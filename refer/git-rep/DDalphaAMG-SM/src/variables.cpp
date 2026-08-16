#include "variables.h"

double pi=3.14159265359;
c_double I_number(0, 1); //imaginary number

long long int FLOPS=0;
long long int localFLOPS=0;

namespace iter_counters{
    long long int kCycleIt=-1;
    long long int vCycleIt=-1;
    long long int CGIt=-1;
    long long int BiCGIt=-1;
    long long int GMRESIt=-1;
    long long int SAPIt=-1;
    long long int FGMRES_SAPIt=-1;
}
namespace flop_counters{
    long long int kCycleFlops=-1;
    long long int vCycleFlops=-1;
    long long int kCycleSetUpFlops=-1;
    long long int vCycleSetUpFlops=-1;
    long long int CGFlops=-1;
    long long int BiCGFLops=-1;
    long long int GMRESFlops=-1;
    long long int SAPFlops=-1;
    long long int FGMRES_SAPFlops=-1;

}

MPI_Datatype global_conf_type;
MPI_Datatype global_conf_resized;
MPI_Datatype local_conf_type;
MPI_Datatype local_conf_resized;
MPI_Datatype column_type;
MPI_Datatype local_domain;
MPI_Datatype local_domain_resized;
MPI_Datatype coarse_domain;
MPI_Datatype coarse_domain_resized;

/*
	Vectorized lattice coords.*/
int Coords(const int& x, const int& t){
	return x*mpi::width_t + t;
}

namespace mass{double m0;}


namespace mpi{
    int rank = 0;
    int size = 1; 
    int maxSize = LV::Ntot; //Default value, will be updated in main
    int maxSizeH = LV::Ntot;
    int sitesH = LV::Ntot;
    int ranks_x = 1;
    int ranks_t = 1;
    int width_x = LV::Nx;
    int width_t = LV::Nt;
    int rank2d = 0; //linearize rank
    int coords[2] = {0,0}; //rank coords
    int top = 0; 
    int bot = 0; 
    int right = 0; 
    int left = 0;
    int bot_left = 0;
    int bot_right = 0;
    int top_left = 0;
    int top_right = 0;
    MPI_Comm cart_comm;

    MPI_Group cart_comm_group;                   
    MPI_Group* coarse_group  = nullptr;  
    MPI_Comm* coarse_comm   = nullptr;   
    MPI_Comm comm_coarse_level = MPI_COMM_NULL;
    int top_c = 0;                                     
    int bot_c = 0; 
    int right_c = 0; 
    int left_c = 0; 
    int coarse_rank2d =-1; //For ranks which don't work on the coarse level
    int ranks_x_c = 1;                              
    int ranks_t_c = 1;                               
    int size_c = 1;                                             
    
    int* rank_dictionary = nullptr;
    MPI_Datatype* column_type = nullptr;
    
}

namespace CG{
	int max_iter = 10000; //Maximum number of iterations for the conjugate gradient method
	double tol = 1e-12; //Tolerance for convergence
}

namespace BiCG{
    int max_iter = 10000; //Maximum number of iterations for the conjugate gradient method
    double tol = 1e-12; //Tolerance for convergence
}

namespace SAPV {
    int sap_gmres_restart_length    = 5;        //GMRES restart length for the Schwarz blocks. 
    int sap_gmres_restarts          = 5;        //GMRES iterations for the Schwarz blocks.
    double sap_gmres_tolerance      = 1e-3;     //GMRES tolerance for the Schwarz blocks
    double sap_tolerance            = 1e-12;    //Tolerance for the SAP method when used as stand-alone solver
}


namespace AMGV {
    int SAP_test_vectors_iterations = 4; //Number of SAP iterations to smooth test vectors
    //Parameters for the coarse level solver. They can be changed in the main function
    int gmres_restarts_coarse_level = 10; 
    int gmres_restart_length_coarse_level = 400; //GMRES restart length for the coarse level
    double gmres_tol_coarse_level = 0.1; //GMRES tolerance for the coarse level

    int nu1 = 0; //Pre-smoothing iterations
    int nu2 = 2; //Post-smoothing iterations
    int Nit = 1; //Number of bootstrap iterations for improving the interpolator

    bool SetUpDone = false; //Set to true when the setup is done, do not change

    //Parameters for FGMRES used in the k-cycle
    int fgmres_k_cycle_restart_length = 5;
    int fgmres_k_cycle_restarts = 2;
    double fgmres_k_cycle_tol = 0.1;

    int cycle = 0; //Cycling stratey. Cycle = 0 -> V-cycle, = 1 --> K-cycle
}

//--------------Parameters for outer FGMRES solver--------------//
namespace FGMRESV {
    double fgmres_tolerance = 1e-10;    //Tolerance for FGMRES
    int fgmres_restart_length = 20;     //Restart length for FGMRES
    int fgmres_restarts = 50;           //Number of restarts for FGMRES
}

namespace LevelV{
    int levels;   //Number of levels
    int maxLevel; //Maximum level id is levels - 1b

    int* BlocksX    = nullptr; //Number of lattice blocks on the x direction for aggregates
    int* BlocksT    = nullptr; //Number of lattice blocks on the t direction for aggregates
    int* Ntest      = nullptr; //Number of test vectors
    int* Nagg       = nullptr; //Number of aggregates
    int* NBlocks    = nullptr; //Number of lattice blocks  
    
    int* Nsites     = nullptr; //Number of sites per rank (no Halo)
    int* NxSites    = nullptr; //Nx sites per rank (no Halo)
    int* NtSites    = nullptr; //Nt sites per rank (no Halo)

    int* NsitesH    = nullptr; //Number of sites per rank (with halo)
    int* NxSitesH   = nullptr; //Nx sites per rank (with halo))
    int* NtSitesH   = nullptr; //Nt sites per rank (with halo)

    int* DOF        = nullptr; //Number of degrees of freedom at each lattice site.
                                //On the finest level, DOF = 2 (only spin), on the coarse levels DOF = 2*Ntest

    int* Colors     = nullptr; //Number of "colors" at each level 
    

    int* SAP_Block_x                = nullptr;  //Number of SAP blocks on the x direction 
    int* SAP_Block_t                = nullptr;  //Number of SAP blocks on the t direction
    int* SAP_elements_x             = nullptr;  //Number of SAP blocks on the x direction 
    int* SAP_elements_t             = nullptr;  //Number of SAP blocks on the t direction
    int* SAP_variables_per_block    = nullptr;  //Number of variables in each SAP block

    int* GMRES_restart_len  = nullptr; 
    int* GMRES_restarts     = nullptr; 
    double* GMRES_tol       = nullptr;

    int* RanksX = nullptr;
    int* RanksT = nullptr;
    MPI_Comm* D_operator_communicator = nullptr;    //Communicator for D_operator at level l
}



int* lpb = nullptr;
int* rpb = nullptr;
c_double* lsign = nullptr;
c_double* rsign = nullptr;

void allocate_lattice_arrays() {
    lpb     = new int[mpi::maxSizeH]();
    rpb     = new int[mpi::maxSizeH]();
    lsign   = new c_double[mpi::maxSizeH]();
    rsign   = new c_double[mpi::maxSizeH]();

    LevelV::BlocksX     = new int[LevelV::maxLevel];
    LevelV::BlocksT     = new int[LevelV::maxLevel];
    LevelV::Ntest       = new int[LevelV::maxLevel];
    LevelV::Nagg        = new int[LevelV::maxLevel];
    LevelV::NBlocks     = new int[LevelV::maxLevel];

    LevelV::Nsites      = new int[LevelV::levels];
    LevelV::NxSites     = new int[LevelV::levels];
    LevelV::NtSites     = new int[LevelV::levels];
    LevelV::NsitesH     = new int[LevelV::levels];
    LevelV::NxSitesH    = new int[LevelV::levels];
    LevelV::NtSitesH    = new int[LevelV::levels];
    LevelV::DOF         = new int[LevelV::levels];
    LevelV::Colors      = new int[LevelV::levels];

    LevelV::SAP_Block_x             = new int[LevelV::levels];
    LevelV::SAP_Block_t             = new int[LevelV::levels];
    LevelV::SAP_elements_x          = new int[LevelV::levels];
    LevelV::SAP_elements_t          = new int[LevelV::levels];
    LevelV::SAP_variables_per_block = new int[LevelV::levels];

    LevelV::GMRES_restart_len   = new int[LevelV::levels];
    LevelV::GMRES_restarts      = new int[LevelV::levels];
    LevelV::GMRES_tol           = new double[LevelV::levels];

    LevelV::RanksX = new int[LevelV::levels];
    LevelV::RanksT = new int[LevelV::levels];
    LevelV::D_operator_communicator = new MPI_Comm[LevelV::levels];

    mpi::coarse_group       = new MPI_Group[mpi::ranks_coarse_level];
    mpi::coarse_comm        = new MPI_Comm[mpi::ranks_coarse_level];
    mpi::rank_dictionary    = new int[mpi::size];
    mpi::column_type        = new MPI_Datatype[LevelV::levels];
}

void free_lattice_arrays() {
    delete[] lpb;
    delete[] rpb;
    delete[] lsign;
    delete[] rsign;

    delete[] LevelV::BlocksX;     
    delete[] LevelV::BlocksT;     
    delete[] LevelV::Ntest;       
    delete[] LevelV::Nagg;        
    delete[] LevelV::NBlocks;     

    delete[] LevelV::Nsites;         
    delete[] LevelV::NxSites;        
    delete[] LevelV::NtSites;        
    delete[] LevelV::NsitesH;        
    delete[] LevelV::NxSitesH;       
    delete[] LevelV::NtSitesH;       
    delete[] LevelV::DOF;            
    delete[] LevelV::Colors;          
    
    delete[] LevelV::SAP_Block_x;             
    delete[] LevelV::SAP_Block_t;           
    delete[] LevelV::SAP_elements_x;          
    delete[] LevelV::SAP_elements_t;          
    delete[] LevelV::SAP_variables_per_block; 

    delete[] LevelV::GMRES_restart_len;  
    delete[] LevelV::GMRES_restarts;    
    delete[] LevelV::GMRES_tol;  

    delete[] LevelV::RanksX;
    delete[] LevelV::RanksT;  
    delete[] LevelV::D_operator_communicator;

    delete[] mpi::coarse_group;
    delete[] mpi::coarse_comm;
    delete[] mpi::rank_dictionary;
    
    delete[] mpi::column_type;
}
