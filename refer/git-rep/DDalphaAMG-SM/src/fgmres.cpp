#include "fgmres.h"
#include "iomanip"

//------Class FGMRES implementation------//
int FGMRES::fgmres(const spinor& phi, const spinor& x0, spinor& x, const bool& print_message) { 
    setZeros();
    int k = 0; //Iteration number (restart cycle)
    x = x0; //initial solution 
    func(x, Dx); //Matrix-vector operation
    axpy(phi,Dx, -1.0, r); //r = b - A*x
	double norm_phi = sqrt(std::real(dot(phi.val, phi.val))); //norm of the right hand side
    double err = sqrt(std::real(dot(r.val, r.val))); //Initial error
    localFLOPS += 2*dsq;
    int maxIt = m;
    int index;
    while (k < restarts) {
        beta = err + 0.0 * I_number;
        localFLOPS += dd+dcm;
        scal(1.0/beta, r,VmT[0]); //VmT[0] = r / ||r||
        localFLOPS += dcd; //Division 1/beta
        gm[0] = beta; //gm[0] = ||r||
        //-----Arnoldi process to build the Krylov basis and the Hessenberg matrix-----//
        for (int j = 0; j < m; j++) {
            preconditioner(VmT[j], ZmT[j]); //ZmT[j] = M^-1 VmT[j]
        
            func(ZmT[j],w); 
            //Gram-Schmidt process to orthogonalize the vectors
            for (int i = 0; i <= j; i++) {
                Hm[i][j] = dot(w.val, VmT[i].val); //  (v_i^dagger, w)
                //w = w -  Hm[i][j] * VmT[i];
                for(int nx = x_ini; nx<=x_fin; nx++){
                for(int nt = t_ini; nt<=t_fin; nt++){
                for(int mu=0; mu<dofs; mu++){
                    index = dofs*(nx*(Nt+2)+nt)+mu;
					w.val[index] -= Hm[i][j] * VmT[i].val[index];
                    localFLOPS += ca+cm;
				}
				}
                }
            }

            Hm[j + 1][j] = sqrt(std::real(dot(w.val, w.val))); //H[j+1][j] = ||A v_j||
            localFLOPS += dsq;
            if (std::real(Hm[j + 1][j]) > 0) {
                scal(1.0 / Hm[j + 1][j], w, VmT[j + 1]); //VmT[j + 1] = w / ||A v_j||
                localFLOPS += dcd;
            }
            //----Rotate the matrix----//
            rotation(j);

            //Rotate gm
            gm[j + 1] = -sn[j] * gm[j];
            gm[j] = std::conj(cn[j]) * gm[j];
            localFLOPS += 2*cm;
            if (std::abs(gm[j+1]) < tol* norm_phi){
                maxIt = j+1;
                break;
            }
            //std::cout << "residual " << std::abs(gm[j+1]) << std::endl;
        }        
        //Solve the upper triangular system//
		solve_upper_triangular(Hm, gm,maxIt,eta);
        
        for(int nx = x_ini; nx<=x_fin; nx++){
        for(int nt = t_ini; nt<=t_fin; nt++){
        for(int mu=0; mu<dofs; mu++){
            index = dofs*(nx*(Nt+2)+nt)+mu;
            for (int j = 0; j < maxIt; j++) {
                x.val[index] = x.val[index] + eta[j] * ZmT[j].val[index]; 
                localFLOPS += ca+cm;
            }
        }
        }
        }
        
        //Compute the residual
        func(x, Dx);
        axpy(phi,Dx, -1.0, r); //r = b - A*x
        
        
        err = sqrt(std::real(dot(r.val, r.val)));
        localFLOPS += dsq;
        //Checking the residual evolution
        if (err < tol* norm_phi) {
            if (print_message == true && mpi::rank2d == 0){ 
                std::cout << "FGMRES converged in " << k + 1 << " cycles" << " Error " << err << std::endl;
                std::cout << "With " << k*m + maxIt  << " iterations" <<  std::endl;
            }
            return k*m + maxIt;
        }
        k++;
    }
    if (print_message == true && mpi::rank2d == 0) 
        std::cout << "FGMRES did not converge in " << restarts << " restarts of lenght " << m << " Error " << err << std::endl;
    
    return restarts*m;
}

void FGMRES::rotation(const int& j) {
    //Rotation of the column elements that are <j
    c_double temp;
    for (int i = 0; i < j; i++) {
		temp = std::conj(cn[i]) * Hm[i][j] + std::conj(sn[i]) * Hm[i + 1][j];
		Hm[i + 1][j] = -sn[i] * Hm[i][j] + cn[i] * Hm[i + 1][j];
		Hm[i][j] = temp;
        localFLOPS += (ca+2*cm)*2;
    }
    //Rotation of the diagonal and element right below the diagonal
    c_double den = sqrt(std::real(std::conj(Hm[j][j] ) * Hm[j][j] + std::conj(Hm[j + 1][j]) * Hm[j + 1][j]));
	sn[j] = Hm[j + 1][j] / den; cn[j] = Hm[j][j] / den;
	Hm[j][j] = std::conj(cn[j]) * Hm[j][j] + std::conj(sn[j]) * Hm[j + 1][j];
    localFLOPS += dsq+ca+2*cm  +  2*cd  +  ca+2*cm;
    Hm[j + 1][j] = 0.0;

}

//x = A^-1 b, A an upper triangular matrix of dimension n
void FGMRES::solve_upper_triangular(const c_matrix& A, const c_vector& b, const int& n, c_vector& out) {
	for (int i = n - 1; i >= 0; i--) {
		out[i] = b[i];
		for (int j = i + 1; j < n; j++) {
			out[i] -= A[i][j] * out[j];
            localFLOPS += ca+cm;
		}
		out[i] /= A[i][i];
        localFLOPS += cd;
	}
}