#include "conjugate_gradient.h"


int conjugate_gradient(const spinor& U, const spinor& phi, spinor& sol, const double& m0,const bool print){
    using namespace mpi;
    int k = 0; //Iteration number
    double err;
    double err_sqr;

    spinor r(mpi::maxSizeH);  //residual
    spinor d(mpi::maxSizeH);  //search direction
    spinor Ad(mpi::maxSizeH); //DD^dagger*d
  
    c_double alpha, beta;

    sol = phi;
    D_D_dagger_phi(U, sol, Ad, m0); //DD^dagger*x
    
    int n;
    for(int x = 1; x<=width_x; x++){
        for(int t = 1; t<=width_t; t++){
            n = x*(width_t+2)+t;
            r.val[2*n]   = phi.val[2*n]   - Ad.val[2*n]; //mu0
            r.val[2*n+1] = phi.val[2*n+1] - Ad.val[2*n+1]; //mu1
            localFLOPS += 2*ca;
        }
    }

    d = r; //initial search direction
 
    c_double r_norm2 = dot(r.val, r.val);
    
    double phi_norm2 = sqrt(std::real(dot(phi.val, phi.val)));
    localFLOPS += dsq;

    while (k<CG::max_iter) {
        D_D_dagger_phi(U, d, Ad, m0); //DD^dagger*d 
        alpha = r_norm2 / dot(d.val, Ad.val); //alpha = (r_i,r_i)/(d_i,Ad_i)
        localFLOPS += cd;
        for(int x = 1; x<=width_x; x++){
            for(int t = 1; t<=width_t; t++){
                n =  x*(width_t+2)+t;
                //x = x + alpha * d; //x_{i+1} = x_i + alpha*d_i 
                sol.val[2*n]    += alpha*d.val[2*n];
                sol.val[2*n+1]  += alpha*d.val[2*n+1];
                //r = r - alpha * Ad; //r_{i+1} = r_i - alpha*Ad_i
                r.val[2*n]      -= alpha*Ad.val[2*n];
                r.val[2*n+1]    -= alpha*Ad.val[2*n+1];

                localFLOPS += 4*(ca+cm);
            }
        }
        
        err_sqr = std::real(dot(r.val, r.val)); //err_sqr = (r_{i+1},r_{i+1})
		err = sqrt(err_sqr); // err = sqrt(err_sqr)
        localFLOPS += dsq;
        if (err < CG::tol*phi_norm2) {
            if (mpi::rank2d == 0 && print == true)
                std::cout << "CG for D^+D converged in " << k+1 << " iterations" << " Error " << err << std::endl;
            return k+1;
        }

        beta = err_sqr / r_norm2; //beta = (r_{i+1},r_{i+1})/(r_i,r_i)
        localFLOPS += cd;
        //d_{i+1} = r_{i+1} + beta*d_i 
        for(int x = 1; x<=width_x; x++){
            for(int t = 1; t<=width_t; t++){
                n =  x*(width_t+2)+t;
                d.val[2*n]      *= beta; 
                d.val[2*n+1]    *= beta;
                d.val[2*n]      += r.val[2*n];
                d.val[2*n+1]    += r.val[2*n+1];
                localFLOPS += 2*ca+2*cm;
            }
        }
        r_norm2 = err_sqr;
        k++;
    }

    if (rank2d == 0)
        std::cout << "CG for D^+D did not converge in " << CG::max_iter << " iterations" << " Error " << err << std::endl;
    return 0;
}


//Solves Dx x = phi with the Bi-CGstab method
int bi_cgstab(const spinor& U, const spinor& phi, const spinor& x0, spinor& x, const double& m0, const bool& print_message) {

    int k = 0; //Iteration number
    double err; // ||r||

    spinor r(mpi::maxSizeH);  //r[coordinate][spin] residual
    spinor r_tilde(mpi::maxSizeH);  //r[coordinate][spin] residual
    spinor d(mpi::maxSizeH); //search direction
    spinor s(mpi::maxSizeH);
    spinor t(mpi::maxSizeH);
    spinor Ad(mpi::maxSizeH); //D*d

    c_double alpha, beta, rho_i, omega, rho_i_2;

    x = x0; //initial solution
    spinor Dphi(mpi::maxSizeH); //Temporary spinor for D x
    D_phi(U, x, Dphi, m0);
    axpy(phi,Dphi, -1.0, r); //r = b - A*x
    r_tilde = r;
	double norm_phi = sqrt(std::real(dot(phi.val, phi.val))); //norm of the right hand side
    localFLOPS += dsq;
    int index;
    while (k<BiCG::max_iter) {
        rho_i = dot(r.val, r_tilde.val); //r . r_dagger
        if (k == 0) {
            d = r; //d_1 = r_0
        }
        else {
            beta = alpha * rho_i / (omega * rho_i_2); //beta_{i-1} = alpha_{i-1} * rho_{i-1} / (omega_{i-1} * rho_{i-2})
            localFLOPS += 2*cm+cd;
            //d = r + beta * (d - omega * Ad);
            for(int nx = 1; nx<=mpi::width_x; nx++){
            for(int nt = 1; nt<=mpi::width_t; nt++){
            for(int mu=0; mu<LV::dof; mu++){
                index = idx(nx,nt,mu);
                d.val[index] = r.val[index] + beta * (d.val[index] - omega * Ad.val[index]); //d_i = r_{i-1} + beta_{i-1} * (d_{i-1} - omega_{i-1} * Ad_{i-1})
                localFLOPS += 2*ca+2*cm;
            }
            }
            }
        }

        D_phi(U, d, Ad, m0);  //A d_i 
        alpha = rho_i / dot(Ad.val, r_tilde.val); //alpha_i = rho_{i-1} / (Ad_i, r_tilde)
        localFLOPS += cd;
        
        //s = r - alpha * Ad; //s = r_{i-1} - alpha_i * Ad_i
        for(int nx = 1; nx<=mpi::width_x; nx++){
        for(int nt = 1; nt<=mpi::width_t; nt++){
        for(int mu=0; mu<LV::dof; mu++){
                index = idx(nx,nt,mu);
                s.val[index] = r.val[index] - alpha * Ad.val[index]; //s_i = r_{i-1} - alpha_i * Ad_i
                localFLOPS += ca+cm;
        }
        }
        }

        err = sqrt(std::real(dot(s.val, s.val)));
        localFLOPS += dsq;
        
        if (err < BiCG::tol * norm_phi) {
            axpy(x,d, alpha, x); //x = x + alpha * d;
            if (print_message == true && mpi::rank2d == 0) {
                std::cout << "Bi-CG-stab for D converged in " << k+1 << " iterations" << " Error " << err << std::endl;
            }
            return k+1;
        }
        D_phi(U, s, t,m0);   //A s
        omega = dot(s.val, t.val) / dot(t.val, t.val); //omega_i = t^dagg . s / t^dagg . t
        localFLOPS += cd;
        //r = s - omega * t; 
        axpy(s,t,-omega,r); //r_i = s - omega_i * t
        //x = x + alpha * d + omega * s; 
        for(int nx = 1; nx<=mpi::width_x; nx++){
        for(int nt = 1; nt<=mpi::width_t; nt++){
        for(int mu=0; mu<LV::dof; mu++){
                index = idx(nx,nt,mu);
                x.val[index] = x.val[index] + alpha * d.val[index] + omega * s.val[index]; //x_i = x_{i-1} + alpha_i * d_i + omega_i * s_i
                localFLOPS += 2*ca+2*cm;
        }
        }
        }

        rho_i_2 = rho_i; //rho_{i-2} = rho_{i-1}
        k++;
    }
    if (print_message == true && mpi::rank2d == 0) 
        std::cout << "Bi-CG-stab for D did not converge in " << BiCG::max_iter << " iterations" << " Error " << err << std::endl;
    
    return BiCG::max_iter;
}