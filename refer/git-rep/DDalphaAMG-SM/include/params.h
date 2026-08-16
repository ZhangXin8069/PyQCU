#include <string>
#include <fstream>
#include <sstream>
#include "mpi_setup.h"


/*
    Function for reading the AMG blocks and test vectors for each level
    The parameters file has the following information on each row
    level, block_x, block_t, ntest, sap_block_x, sap_block_t
    -level: multigrid level
    -block_x: number of blocks in the x direction used for the aggregation
    -block_t: number of blocks in the t direction used for the aggregation
    -ntest: number of test vectors to go from level to level + 1
    -sap_block_x, sap_block_t: number of blocks in the x, t directions used for SAP
    We don't need to specify any parameter for the coarsest level, because we don't build a blocking there.
*/
void readParameters(const std::string& inputFile){
    std::ostringstream NameData;
    NameData << inputFile;
    std::ifstream infile(NameData.str());
    if (!infile) {
        std::cerr << "File " << NameData.str() <<  " not found" << std::endl;
        exit(1);
    }
    int block_x, block_t, ntest;
    int level; 
    int maxLevel = LevelV::maxLevel;
    int sap_block_x, sap_block_t;
    LevelV::RanksX[0] = mpi::ranks_x;
    LevelV::RanksT[0] = mpi::ranks_t;
    int count = 1;
    //read parameters for level < max_level
    while (infile >> level >> block_x >> block_t >> ntest >> sap_block_x >> sap_block_t) {
        LevelV::BlocksX[level] = block_x;
        LevelV::BlocksT[level] = block_t;
        LevelV::Ntest[level] = ntest;
        LevelV::Nagg[level] = 2*block_x*block_t;
        LevelV::NBlocks[level] = block_x * block_t;
        LevelV::Nsites[level] = (level == 0 ) ? LV::Ntot : LevelV::BlocksX[level-1] * LevelV::BlocksT[level-1];
        LevelV::NxSites[level] = (level == 0 ) ? LV::Nx : LevelV::BlocksX[level-1];
        LevelV::NtSites[level] = (level == 0 ) ? LV::Nt : LevelV::BlocksT[level-1];
		LevelV::DOF[level] = (level == 0 ) ? 2 : 2 * LevelV::Ntest[level-1];
        LevelV::Colors[level] = (level == 0 ) ? 1 : LevelV::Ntest[level-1]; 
        
        LevelV::SAP_Block_x[level] = sap_block_x;
        LevelV::SAP_Block_t[level] = sap_block_t;
        LevelV::SAP_elements_x[level] = LevelV::NxSites[level]/sap_block_x; 
        LevelV::SAP_elements_t[level] = LevelV::NtSites[level]/sap_block_t; 
        LevelV::SAP_variables_per_block[level] = LevelV::DOF[level] * LevelV::SAP_elements_x[level] * LevelV::SAP_elements_t[level] ; 

        //These GMRES parameters are not really used in AMG. Only If I want to smooth with GMRES... //
        LevelV::GMRES_restart_len[level] = 20;
        LevelV::GMRES_restarts[level] = 20;
        LevelV::GMRES_tol[level] = 1e-10;
        count++;
    }
    //Store the number of sites and degrees of freedom for the coarsest lattice as well
    LevelV::Nsites[maxLevel] =   LevelV::BlocksX[maxLevel-1] * LevelV::BlocksT[maxLevel-1];
	LevelV::DOF[maxLevel] =  2 * LevelV::Ntest[maxLevel-1];
    LevelV::NxSites[maxLevel] = LevelV::BlocksX[maxLevel-1];
    LevelV::NtSites[maxLevel] = LevelV::BlocksT[maxLevel-1];
    LevelV::Colors[maxLevel] = LevelV::Ntest[maxLevel-1];
    LevelV::SAP_Block_x[maxLevel] = 1; //For the coarsest level we don't need a blocking, but it is necessary to define some numbers here.
    LevelV::SAP_Block_t[maxLevel] = 1;

    LevelV::GMRES_restart_len[maxLevel] = 20;
    LevelV::GMRES_restarts[maxLevel] = 20;
    LevelV::GMRES_tol[maxLevel] = 0.1;

    for(level = 1; level<LevelV::levels; level++)
        LevelV::D_operator_communicator[level] = MPI_COMM_NULL;

    LevelV::D_operator_communicator[0] = mpi::cart_comm;
    bool comm_is_cart_comm = true;
    for(level = 0; level<LevelV::levels-1; level++){
        int xranks_per_block  = (level != LevelV::maxLevel) ? LevelV::RanksX[level]/LevelV::BlocksX[level] : 1; //x-ranks inside a block 
        int tranks_per_block  = (level != LevelV::maxLevel) ? LevelV::RanksT[level]/LevelV::BlocksT[level] : 1; //t-ranks inside a block 
        int ranks_per_block   = xranks_per_block*tranks_per_block;  //Number of ranks inside a lattice block
        if (ranks_per_block > 1 || comm_is_cart_comm == false){
            comm_is_cart_comm = false;
            LevelV::RanksX[level+1] = mpi::coarse_ranks_x;
            LevelV::RanksT[level+1] = mpi::coarse_ranks_t;
            if (mpi::comm_coarse_level != MPI_COMM_NULL){
                //std::cout << "coarse comm only exists for rank " << mpi::rank2d << std::endl;
                LevelV::D_operator_communicator[level+1] = mpi::comm_coarse_level;
            }
        }
        else{
            //if (mpi::rank2d == 0)
            //    std::cout << "Level " << level+1 << " preserves the original communicator" << std::endl;
            LevelV::D_operator_communicator[level+1] = mpi::cart_comm;
            LevelV::RanksX[level+1] = mpi::ranks_x;
            LevelV::RanksT[level+1] = mpi::ranks_t;
        }  
    }
   

    infile.close();
    if (mpi::rank2d == 0 && count!=LevelV::levels){
        std::cout << "The number of levels in the inputs file does not match the number of levels assigned on the terminal" << std::endl;
        MPI_Abort(mpi::cart_comm, EXIT_FAILURE);
    }

    if (mpi::rank2d == 0){
        std::cout << "Parameters read from " << NameData.str() << std::endl;
        if (comm_is_cart_comm == true)
            std::cout << "No rank coarsening necessary" << std::endl;
    }
}