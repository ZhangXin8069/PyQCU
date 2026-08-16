#ifndef FGMRES_H
#define FGMRES_H

#include <utility> //for std::move
#include "dirac_operator.h"

typedef std::vector<c_double> c_vector;
typedef std::vector<c_vector> c_matrix; //Useful for the Hessenberg matrix

/*
	FGMRES 
    Ntot: number of lattice sites (without halos)
    dof: number of degrees of freedom per lattice site
    maxSizeH: maximum arrays size. Either Ntot*dof (no halos) or (Nx+2)(Nt+2)dof (with halos)
    x_ini, t_ini, x_fin, t_fin: Limits for looping arrays. This might depend on whether there are halos or not.
    m: restart length
    restarts: number of restarts
    tol: tolerance of the solver

    The subclasses implement the matrix-vector operation and the preconditioner
    The convergence criterion is ||r|| < ||phi|| * tol
*/
class FGMRES{
    public:
    
    FGMRES( const int Ntot, const int dofs, const int maxSizeH,
            const int x_ini, const int t_ini, 
            const int x_fin, const int t_fin,
            const int m, const int restarts, const double tol) : 
            Ntot(Ntot), dofs(dofs), maxSizeH(maxSizeH),
            x_ini(x_ini), x_fin(x_fin),
            t_ini(t_ini), t_fin(t_fin),
            m(m), restarts(restarts), tol(tol) {
        VmT = std::vector<spinor>(m+1,spinor(maxSizeH)); 
        ZmT = std::vector<spinor>(m, spinor(maxSizeH));  //Z matrix transpose
        Hm  = c_matrix(m+1 , c_vector(m, 0)); 
        gm  = c_vector (m + 1, 0); 
        sn  = c_vector (m, 0);
        cn  = c_vector (m, 0);
        eta = c_vector (m, 0);
        r   = spinor(maxSizeH);
        w   = spinor(maxSizeH);
        Dx  = spinor(maxSizeH);  
        Nx = x_fin-x_ini+1;
        Nt = t_fin-t_ini+1;
    }	

	~FGMRES() { };

    int fgmres(const spinor&phi, const spinor& x0, spinor& x, const bool& print_message = false); //1 --> converged, 0 --> not converged

    private:

    const int Ntot; 
    const int dofs; 
    const int x_ini; const int x_fin; 
    const int t_ini; const int t_fin;
    const int m; //Restart length
    const int restarts; //Number of restarts
    const double tol; //Tolerance for the solver
    const int maxSizeH; //Maximum array size (might include halos or not)
    int Nt, Nx;


    spinor r;  //r[coordinate][spin] residual
    //VmT[column vector index][vector arranged in matrix form]
    std::vector<spinor> VmT; //V matrix transpose-->dimensions exchanged
    std::vector<spinor> ZmT;  //Z matrix transpose
    c_matrix Hm; //H matrix (Hessenberg matrix)
    c_vector gm; 

    //Elements of rotation matrix |sn[i]|^2 + |cn[i]|^2 = 1
    c_vector sn;
    c_vector cn;
    //Solution to the triangular system
    c_vector eta;
    spinor w;
    spinor Dx; //auxiliary spinor
    c_double beta; //not 1/g^2 from simulations


    /*
    Matrix-vector operation
    This is defined in the derived classes
    */
    virtual void func(const spinor& in, spinor& out) = 0; 

    /*
    Preconditioner operation
    */
    virtual void preconditioner(const spinor& in, spinor& out) = 0; 

    /*
        Virtual functions for vector operations. 
        They are implemented for each subclass and should consider the correct dimensions
    */
    // A.B*
    virtual c_double dot(c_double* A, c_double* B) = 0;
    // out = X + lambda * Y
    virtual void axpy(const spinor& X, const spinor& Y, const c_double &lambda,  spinor& out) = 0;
    // Y = lambda * X
    virtual void scal(const c_double& lambda, const spinor& X, spinor& Y) = 0;

    /*
    Rotations to transform Hessenberg matrix to upper triangular form
    cn: cosine components of the rotation
    sn: sine components of the rotation
    H: Hessenberg matrix
    j: index of the column being processed
    */
    void rotation(const int& j); 

    /*
    Solves an upper triangular system Ax = b, where A is an upper triangular matrix of dimension n
    A: upper triangular matrix
    b: right-hand side vector
    n: dimension of the matrix
    out: output vector where the solution will be stored
    */
    void solve_upper_triangular(const c_matrix& A, const c_vector& b, const int& n, c_vector& out);

    void setZeros(){
        //We set all these to zero to avoid memory issues
        for(int i = 0; i < m + 1; i++) {
            gm[i] = 0.0; //gm vector
            for(int j = 0; j < maxSizeH; j++) {
                VmT[i].val[j] = 0.0;
                ZmT[i%m].val[j] = 0.0; //ZmT matrix
            }
            for(int j = 0; j < m; j++) {
                Hm[i][j] = 0.0; //Hm matrix
            }
        }
        for(int i = 0; i < m; i++) {
            sn[i] = 0.0; //sn vector
            cn[i] = 0.0; //cn vector
            eta[i] = 0.0; //eta vector
        }

        for(int j = 0; j < maxSizeH; j++) {
            r.val[j] = 0.0; //r vector
            w.val[j] = 0.0; //w vector
            Dx.val[j] = 0.0; //Dx vector
        }
        
    }

};

/*
    FGMRES for the fine level
*/
class FGMRES_fine_level : public FGMRES {
    public:
    FGMRES_fine_level(  const int Ntot, const int dofs, const int maxSizeH,
                        const int x_ini, const int t_ini, 
                        const int x_fin, const int t_fin,
                        const int m, const int restarts, const double tol,
                        const spinor& U, const double& m0) : FGMRES(Ntot,dofs,maxSizeH,
                        x_ini, t_ini, x_fin, t_fin,
                        m, restarts, tol), U(U), m0(m0){
    };
    ~FGMRES_fine_level() { };
    
private:
    const spinor& U; //reference to Gauge configuration. This is to avoid copying the matrix
    const double m0; //reference to mass parameter

    /*
    Implementation of the function that computes the matrix-vector product for the fine level
    */
    void func(const spinor& in, spinor& out) override {
        D_phi(U, in, out, m0);
    }

    void preconditioner(const spinor& in, spinor& out) override {
        //No specific preconditioner needed for the fine level
        out = std::move(in); //Identity operation
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