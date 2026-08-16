#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string> 
#include <random>
#include "variables.h"

//mean of a vector
template <typename T>
double mean(std::vector<T> x){ 
    double prom = 0;
    for (T i : x) {
        prom += i*1.0;
    }   
    prom = prom / x.size();
    return prom;
}

//random double number in the inteval [a,b] a = min, b = max
inline double rand_range(double a, double b){
    double cociente = ((double) rand() / (RAND_MAX));
    double x = (b-a) * cociente + a;
    return x;
}

//----------Jackknife---------//
std::vector<double> samples_mean(std::vector<double> dat, int bin); 
double Jackknife_error(std::vector<double> dat, int bin); 
double Jackknife(std::vector<double> dat, std::vector<int> bins); 

//---------------Linspace (similar to python)----------------------//
template <typename T>
std::vector<double> linspace(T min, T max, int n) {
    std::vector<double> linspace;
    double h = (1.0*max - 1.0*min) / (n - 1);
    for (int i = 0; i < n; ++i) {
        linspace.insert(linspace.begin() + i, min*1.0 + i * h); 
    }
    return linspace;
}


inline int idx(int x, int t, int mu) {
    //x ranges from 0 to width_x+1
    //t ranges from 0 to width_t+1
    //The physical volume runs from 1 to width_x (or width_t)
    //mu = 0, 1
    return ((x*(mpi::width_t+2) + t)*LV::dof + mu);
}


/*
	Modulo operation
*/
inline int mod(int a, int b) {
	int r = a % b;
	return r < 0 ? r + b : r;
}

//dot, axpy and scal defined here only work at the finest level and considering spinors with the halos

/*
    dot product
*/
inline c_double dot(c_double* A, c_double* B) {
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
template <typename T>
inline void axpy(const spinor& X, const spinor& Y, const T&lambda,  spinor& out) {
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
template <typename T>
inline void scal(const T& lambda, const spinor& X, spinor& Y) {
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

//Formats decimal numbers
//Useful for writing m0 and beta on the file name
inline std::string format(const double& number) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << number;
    std::string str = oss.str();
    str.erase(str.find('.'), 1); //Removes decimal dot
    return str;
}


/*
Generate a random U(1) variable
*/
c_double RandomU1(); 

void printParameters();


inline void mpi_reduceFLOPS(){
    FLOPS = 0;
    MPI_Allreduce(&localFLOPS, &FLOPS, 1, MPI_LONG_LONG_INT, MPI_SUM, mpi::cart_comm);
}

inline void printFLOPS(const long long int& x){
    long double y = x*1.0;
    if (mpi::rank2d == 0)
        std::cout << "GFLOPS = " << y/1e9 << std::endl;
}

#endif