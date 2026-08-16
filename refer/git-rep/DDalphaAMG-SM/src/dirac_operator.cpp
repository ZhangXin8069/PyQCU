#include "dirac_operator.h"


/*
 *                t                    2D parallelization
 *   0  +-------------------+  Nt   +---------------------+
 *      |                   |       |  rank 0  |  rank 1  |
 *      |                   |       |---------------------|
 *      |                   |       |  rank 2  |  rank 3  |
 *   x  |                   |       |---------------------|
 *      |                   |       |  rank 4  |  rank 5  |
 *      |                   |       |---------------------|
 *      |                   |       |  rank 6  |  rank 7  |
 *   Nx +-------------------+ Nt    +---------------------+
 *                Nx
 * rpb[2*n+1]	 = x+1, t (towards down)
 * lpb[2*n+1] 	 = x-1, t (towards up)
 * rpb[2*n]   	 = x, t+1 (towards right)
 * lpb[2*n]    	 = x, t-1 (towards left)
 * n = x * Nt + t = (x,t) coordinates
 * 
 */
//Eqs (34) of the documentation
void D_phi(const spinor& U, const spinor&  phi, spinor&  Dphi, const double& m0){
	using namespace LV;
	using namespace mpi;
	MPI_Status status;

	//Communicate halos 
	exchange_halo(phi.val);
	exchange_halo(U.val);
	int n;
	for(int x = 1; x<=width_x; x++){
		for(int t = 1; t<=width_t; t++){
			n = x*(width_t+2)+t;
			//mu = 0
			Dphi.val[2*n] = (m0 + 2) * phi.val[2*n] - 0.5 * ( 
					U.val[2*n] 	 * rsign[2*n]   	* (phi.val[2*rpb[2*n]] - phi.val[2*rpb[2*n]+1])
				+	U.val[2*n+1] * rsign[2*n+1] 	* (phi.val[2*rpb[2*n+1]] + I_number * phi.val[2*rpb[2*n+1]+1])
				+ std::conj(U.val[2*lpb[2*n]]) 		* lsign[2*n] 	* (phi.val[2*lpb[2*n]] + phi.val[2*lpb[2*n]+1])
				+ std::conj(U.val[2*lpb[2*n+1]+1]) 	* lsign[2*n+1]  *  (phi.val[2*lpb[2*n+1]] - I_number*phi.val[2*lpb[2*n+1]+1])
			);
			//mu = 1
			Dphi.val[2*n+1] = (m0 + 2) * phi.val[2*n+1] - 0.5 * ( 
					U.val[2*n] 	 * rsign[2*n] 		* (-phi.val[2*rpb[2*n]] + phi.val[2*rpb[2*n]+1])
				+	U.val[2*n+1] * rsign[2*n+1] 	* (-I_number*phi.val[2*rpb[2*n+1]] + phi.val[2*rpb[2*n+1]+1])
				+ std::conj(U.val[2*lpb[2*n]]) 		* lsign[2*n] 	* (phi.val[2*lpb[2*n]] + phi.val[2*lpb[2*n]+1])
				+ std::conj(U.val[2*lpb[2*n+1]+1]) 	* lsign[2*n+1]  * (I_number*phi.val[2*lpb[2*n+1]] + phi.val[2*lpb[2*n+1]+1])
			);
		}
	}
	localFLOPS += 81*2*width_x*width_t;
		
}

void D_dagger_phi(const spinor&  U, const spinor&  phi, spinor&  Dphi, const double& m0){
	using namespace LV;
	using namespace mpi;
	MPI_Status status;

	//Communicate halos 
	exchange_halo(phi.val);
	exchange_halo(U.val);
	int n;
	for(int x = 1; x<=width_x; x++){
		for(int t = 1; t<=width_t; t++){
			n = x*(width_t+2)+t;
			//mu = 0
			Dphi.val[2*n] = (m0 + 2) * phi.val[2*n] -0.5 * ( 
				std::conj(U.val[2*lpb[2*n]]) 		* lsign[2*n] 	* (phi.val[2*lpb[2*n]] - phi.val[2*lpb[2*n]+1])
			+   std::conj(U.val[2*lpb[2*n+1]+1]) 	* lsign[2*n+1] 	* (phi.val[2*lpb[2*n+1]] + I_number * phi.val[2*lpb[2*n+1]+1])
			+   U.val[2*n] 		* rsign[2*n] 		* (phi.val[2*rpb[2*n]] + phi.val[2*rpb[2*n]+1])
			+	U.val[2*n+1] 	* rsign[2*n+1] 		* (phi.val[2*rpb[2*n+1]] - I_number * phi.val[2*rpb[2*n+1]+1])
			);
			//mu = 1
			Dphi.val[2*n+1] = (m0 + 2) * phi.val[2*n+1] -0.5 * ( 
				std::conj(U.val[2*lpb[2*n]]) 		* lsign[2*n] 	* (-phi.val[2*lpb[2*n]] + phi.val[2*lpb[2*n]+1])
			+   std::conj(U.val[2*lpb[2*n+1]+1]) 	* lsign[2*n+1] 	* (-I_number*phi.val[2*lpb[2*n+1]] + phi.val[2*lpb[2*n+1]+1])
			+   U.val[2*n] 		* rsign[2*n] 		* (phi.val[2*rpb[2*n]] + phi.val[2*rpb[2*n]+1])
			+	U.val[2*n+1] 	* rsign[2*n+1] 		* (I_number * phi.val[2*rpb[2*n+1]] + phi.val[2*rpb[2*n+1]+1])
			);
		}
	}	
	localFLOPS += 81*2*width_x*width_t;
}

//D^dagger D phi
void D_D_dagger_phi(const spinor&  U, const spinor&  phi, spinor&  Dphi, const double& m0){
	spinor ddagg_buffer(mpi::maxSizeH);
	D_phi(U, phi, ddagg_buffer, m0);
	D_dagger_phi(U,  ddagg_buffer, Dphi, m0);
}