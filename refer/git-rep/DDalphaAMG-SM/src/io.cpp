#include "io.h"


void read_binary(const std::string& name,const spinor& U){
    using namespace LV;
    std::ifstream infile(name, std::ios::binary);
    if (!infile) {
       std::cerr << "File " << name << " not found " << std::endl;
        exit(1);
    }
    spinor GlobalConf((Nt+2)*(Nx+2)*2); //Temporary variable to store the full configuration


    int counts[mpi::size];
    int displs[mpi::size];
    for(int r = 0; r < mpi::size; r++){
        counts[r] = 1;
        int rx = r / mpi::ranks_t; 
        int rt = r % mpi::ranks_t; 

        // Global starting position inside the buffer including halo (halo at index 0)
        int global_x_start = rx * mpi::width_x + 1; // +1 to skip halo
        int global_t_start = rt * mpi::width_t + 1; // +1 to skip halo
        // Displacement in complex-element units into buffer.val (including halo padding)
        displs[r] = (global_x_start * (LV::Nt + 2) + global_t_start) * 2;

        //if (mpi::rank2d == 0)
        //    std::cout << "displ " << displs[r] << std::endl;
    }

    if (mpi::rank2d == 0){
        for (int x = 1; x <= LV::Nx; x++) {
        for (int t = 1; t <= LV::Nt; t++) {
            int n = x * (LV::Nt+2) + t;
            for (int mu = 0; mu < 2; mu++) {
                int x_read, t_read, mu_read;
                double re, im;
                infile.read(reinterpret_cast<char*>(&x_read), sizeof(int));
                infile.read(reinterpret_cast<char*>(&t_read), sizeof(int));
                infile.read(reinterpret_cast<char*>(&mu_read), sizeof(int));
                infile.read(reinterpret_cast<char*>(&re), sizeof(double));
                infile.read(reinterpret_cast<char*>(&im), sizeof(double));
                GlobalConf.val[2*n+mu] = c_double(re, im); 
            }
        }
        }
        infile.close();
        std::cout << "Binary conf read from " << name << std::endl;     
    }

    int input_ini_local = 2 * (mpi::width_t + 2 + 1);
    MPI_Scatterv(&GlobalConf.val[0], counts, displs, global_conf_resized, &U.val[input_ini_local],1,
        	local_conf_resized, 0, mpi::cart_comm);

}


void broadcast_file_name(std::string& File){
    //Broadcast file name from rank 0 to other ranks
    int filename_len = 0;
    if (mpi::rank == 0) 
        filename_len = static_cast<int>(File.size());

    MPI_Bcast(&filename_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (mpi::rank != 0) 
        File.resize(filename_len);
    
    MPI_Bcast(File.data(), filename_len, MPI_CHAR, 0, MPI_COMM_WORLD);
}

void writeMetadata(int cycle,double tol){
    if (mpi::rank2d == 0){
        std::ostringstream FileName;
        FileName << "metadata_" << LV::Nx << "x" << LV::Nt << "_Levels" << LevelV::levels; 
        if (cycle == 0)
            FileName << "_Vcycle"; 
        else 
            FileName << "_Kcycle";
        FileName << ".dat";
    
        std::ofstream metadata(FileName.str());
        if (!metadata.is_open()) {
            std::cerr << "Error opening naming metadata_NXxNT_Levels_Ntest_Cycle.dat for writing." << std::endl;
            return ;
        } 

        std::string start_time_str;
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        std::ostringstream tss;
        // format: YYYY-MM-DD HH:MM:SS 
        tss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");
        start_time_str = tss.str();

        metadata << "Host: " << std::getenv("HOSTNAME") << "\n";
        metadata << "Date: " << start_time_str << "\n";
        metadata << "x ranks: " << mpi::ranks_x << "  t_ranks: " << mpi::ranks_t << "\n";
        metadata << "m0: " << std::setprecision(7) << mass::m0 << "\n";
        metadata << "solvers tolerance: " << std::setprecision(4) << tol << "\n";
        metadata << "#level, blocks_x (global), blocks_t (global), Ntest, SAP_blocks_x (local), SAP_blocks_t (local)\n";
        for (int l = 0; l < LevelV::levels-1; l++) {
            metadata << l << std::setw(5) << LevelV::BlocksX[l] << std::setw(5) << LevelV::BlocksT[l] << std::setw(5) 
                     << LevelV::Ntest[l]  << std::setw(5) << LevelV::SAP_Block_x[l] << std::setw(5) << LevelV::SAP_Block_t[l] << "\n"; 
        }
        metadata << "#SAP_test_vectors_iterations\n";
        metadata << AMGV::SAP_test_vectors_iterations << "\n"; //Number of smoothing iterations for the test vectors
        metadata << "#cycle (0 = V, 1 = K)\n";
        metadata <<  cycle << "\n"; //Cycle type
        metadata << "#Pre-smoothing nu1, Post-smoothing nu2\n";
        metadata << AMGV::nu1 << std::setw(5) << AMGV::nu2 << "\n"; //Pre and post smoothing iterations
        metadata << "#sap gmres restart length, sap gmres restarts, sap gmres tolerance\n";
        metadata << SAPV::sap_gmres_restart_length << std::setw(5) << SAPV::sap_gmres_restarts << std::setw(15) 
                 << std::scientific << std::setprecision(7) << SAPV::sap_gmres_tolerance << "\n";
        metadata << "#GMRES restart length coarse level, GMRES restarts coarse level, GMRES tolerance coarse level\n";
        metadata << AMGV::gmres_restart_length_coarse_level << std::setw(5) <<AMGV::gmres_restarts_coarse_level << std::setw(15) 
                 << std::scientific << std::setprecision(7) << AMGV::gmres_tol_coarse_level << "\n";
        metadata << "#FGMRES restart length, FGMRES restarts, FGMRES tolerance\n";
        metadata << FGMRESV::fgmres_restart_length << std::setw(5) << FGMRESV::fgmres_restarts << std::setw(15) 
                 << std::scientific << std::setprecision(7) << FGMRESV::fgmres_tolerance << "\n";
        
        //Iterations and flops
        metadata << "#CG iterations,   FLOPS\n";
        metadata << iter_counters::CGIt << std::setw(30) << flop_counters::CGFlops << "\n";
        metadata << "#BiCGStab iterations,   FLOPS\n";
        metadata << iter_counters::BiCGIt << std::setw(30) << flop_counters::BiCGFLops << "\n";
        metadata << "#GMRES iterations,   FLOPS\n";
        metadata << iter_counters::GMRESIt << std::setw(30) << flop_counters::GMRESFlops << "\n";
        metadata << "#SAP iterations,   FLOPS\n";
        metadata << iter_counters::SAPIt << std::setw(30) << flop_counters::SAPFlops << "\n";
        metadata << "#FGMRES_SAP iterations,   FLOPS\n";
        metadata << iter_counters::FGMRES_SAPIt << std::setw(30) << flop_counters::FGMRES_SAPFlops << "\n";
        if (cycle == 0){
            metadata << "#V cycle fgmres iterations,  Setup phase FLOPS, Total FLOPS\n";
            metadata << iter_counters::vCycleIt << std::setw(30) << flop_counters::vCycleSetUpFlops << std::setw(30) << flop_counters::vCycleFlops << "\n";
        } 
        else{
            metadata << "#K cycle fgmres iterations,  Setup phase FLOPS, Total FLOPS\n";
            metadata << iter_counters::kCycleIt << std::setw(30) << flop_counters::kCycleSetUpFlops << std::setw(30) << flop_counters::kCycleFlops << "\n";
        }
    
        metadata.close();

    }

}