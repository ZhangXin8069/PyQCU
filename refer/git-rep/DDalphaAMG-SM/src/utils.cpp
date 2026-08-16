#include "utils.h"

//----------Jackknife---------//
std::vector<double> samples_mean(std::vector<double> dat, int bin) {
    std::vector<double> samples_mean(bin);
    int dat_bin = dat.size() / bin;
    double prom = 0;
    for (int i = 0; i < bin; i++) {
        for (int k = 0; k < bin; k++) {
            for (int j = k * dat_bin; j < k * dat_bin + dat_bin; j++) {
                if (k != i) {
                    prom += dat[j];
                }
            }
        }
        prom = prom / (dat.size() - dat_bin);
        samples_mean[i] = prom;
        prom = 0;
    }
    return samples_mean;
}

double Jackknife_error(std::vector<double> dat, int bin) {
    double error = 0;
    std::vector<double> sm = samples_mean(dat, bin);
    double normal_mean = mean(dat);
    for (int m = 0; m < bin; m++) {
        error += pow((sm[m] - normal_mean), 2);
    }
    error = sqrt(error * (bin - 1) / bin);
    return error;
}


double Jackknife(std::vector<double> dat, std::vector<int> bins) {
    std::vector<double> errores(bins.size());
    double error;
    for (int i = 0; i < bins.size(); i++) {
        errores[i] = Jackknife_error(dat, bins[i]);
    }
    error = *std::max_element(errores.begin(), errores.end());
    return error;
}
//-------------End of Jackknife--------------//

c_double RandomU1() {
	//Random angle in (0,2*pi) with uniform distribution 
	double cociente = ((double) rand() / (RAND_MAX));
    double theta = 2.0*pi * cociente;
	c_double z(cos(theta), sin(theta));
	return z;
}


void printParameters(){
    if (mpi::rank2d == 0){
    using namespace LV; //Lattice parameters namespace
        std::cout << "******************* Parameters summary *******************" << std::endl;
        std::cout << "| Nx = " << Nx << " Nt = " << Nt << std::endl;
        std::cout << "| Lattice dimension = " << (Nx * Nt) << std::endl;
        std::cout << "| Number of entries of the Dirac matrix = (" << (2 * Nx * Nt) << ")^2" << std::endl;
        std::cout << "| Bare mass m0 = " << mass::m0 << std::endl;
        std::cout << "| Ranks_x = " << mpi::ranks_x << "  Ranks_t = " << mpi::ranks_t << std::endl;
        std::cout << "---------------------------------------------------------------------------------------" << std::endl;  
        std::cout << "* Blocks, aggregates and test vectors at each level" << std::endl;
        using namespace LevelV; //Lattice parameters namespace
        for(int l=0; l< levels-1; l++){
            std::cout << "| Level " << l << " Block X " << BlocksX[l] 
            << " Block T " << BlocksT[l] << " Ntest " << Ntest[l] << " Nagg " << Nagg[l]
            << " Number of lattice blocks " << NBlocks[l] 
            << " Schwarz Block T " << SAP_Block_t[l] << " Schwarz Block X " << SAP_Block_x[l] << std::endl;
        }
        std::cout << "---------------------------------------------------------------------------------------" << std::endl;
        std::cout << "* SAP blocks per rank" << std::endl;
        for(int l=0; l< levels-1; l++){
            std::cout << "| Level " << l << " Schwarz Block T " << SAP_Block_t[l] << " Schwarz Block X " << SAP_Block_x[l]  
            << " Number of blocks " << SAP_Block_t[l]*SAP_Block_x[l] << std::endl;
        }
        std::cout << "---------------------------------------------------------------------------------------" << std::endl;
        std::cout << "* Sites and degrees of freedom at each level" << std::endl;
        for(int l=0; l< levels; l++){
            std::cout << "| Level " << l << " Nsites " << Nsites[l] 
            << " Nxsites " << NxSites[l] << " NtSites " << NtSites[l] << " DOF " << DOF[l]
            << " Colors " << Colors[l] << std::endl;
        }
        std::cout << "---------------------------------------------------------------------------------------" << std::endl;
        std::cout << "| GMRES restart length for SAP blocks = " << SAPV::sap_gmres_restart_length << std::endl;
        std::cout << "| GMRES restarts for SAP blocks = " << SAPV::sap_gmres_restarts << std::endl;
        std::cout << "| GMRES tolerance for SAP blocks = " << SAPV::sap_gmres_tolerance << std::endl;
        std::cout << "---------------------------------------------------------------------------------------" << std::endl;
        std::cout << "| Number of levels = " << levels << std::endl;
        std::cout << "| nu1 (pre-smoothing) = " << AMGV::nu1 << " nu2 (post-smoothing) = " << AMGV::nu2 << std::endl;
        std::cout << "| Number of iterations for improving the interpolator = " << AMGV::Nit << std::endl;
        std::cout << "| Restart length of GMRES at the coarse level = " << AMGV::gmres_restart_length_coarse_level << std::endl;
        std::cout << "| Restarts of GMRES at the coarse level = " << AMGV::gmres_restarts_coarse_level << std::endl;
        std::cout << "| GMRES tolerance for the coarse level solution = " << AMGV::gmres_tol_coarse_level << std::endl;
        std::cout << "* FGMRES with AMG preconditioning parameters" << std::endl;
        std::cout << "| FGMRES restart length = " << FGMRESV::fgmres_restart_length << std::endl;
        std::cout << "| FGMRES restarts = " << FGMRESV::fgmres_restarts << std::endl;
        std::cout << "| FGMRES tolerance = " << FGMRESV::fgmres_tolerance << std::endl;
        std::cout << "*****************************************************************************************************" << std::endl;
    }
}
