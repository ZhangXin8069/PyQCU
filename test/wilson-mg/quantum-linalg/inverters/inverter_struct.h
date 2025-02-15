// Copyright (c) 2017 Evan S Weinberg
// Include file for the inverter info struct.

#ifndef QLINALG_INVERTER_STRUCT
#define QLINALG_INVERTER_STRUCT

#include <iostream>
using namespace std;

// Struct that contains information that
// all matrix functions return.
// Return for various matrix functions.
struct inversion_info
{
  double resSq; // squared residual.
  int iter; // number of iterations.
  bool success; // did we reach residual?
  std::string name; // name of algorithm.
  int ops_count; // how many times was the matrix op called?
    
  double* resSqmrhs; // squared residual. Only assigned for multirhs solves. 
  int n_rhs; 
  
  // Default Constructor
  inversion_info() : resSq(0.0), iter(0), success(false), name(""), ops_count(0), resSqmrhs(0), n_rhs(-1)
  { }
  
  // Multirhs constructor
  inversion_info(int in_n_rhs) : resSq(0.0), iter(0), success(false), name(""), ops_count(0), resSqmrhs(0), n_rhs(in_n_rhs)
  {
    resSqmrhs = new double[in_n_rhs];
  }
  
  // Destructor.
  ~inversion_info()
  {
    if (resSqmrhs != 0 && n_rhs > 0)
    {
      delete[] resSqmrhs;
      resSqmrhs = 0;
    }
  }
  
  // Copy constructor.
  inversion_info(const inversion_info &obj)
  {
    resSq = obj.resSq;
    iter = obj.iter;
    success = obj.success;
    name = obj.name;
    ops_count = obj.ops_count;
    
    n_rhs = obj.n_rhs;
    if (obj.resSqmrhs != 0 && n_rhs > 0)
    {
      resSqmrhs = new double[n_rhs];
      for (int i = 0; i < n_rhs; i++)
      {
        resSqmrhs[i] = obj.resSqmrhs[i];
      }
    }
    else
    {
      resSqmrhs = 0;
    }
  }
  
  // Assignment operator.
  inversion_info& operator=(inversion_info obj) // Pass by value (thus generating a copy)
  {
    resSq = obj.resSq;
    iter = obj.iter;
    success = obj.success;
    name = obj.name;
    ops_count = obj.ops_count;
    
    n_rhs = obj.n_rhs;
    if (obj.resSqmrhs != 0 && n_rhs > 0)
    {
      resSqmrhs = obj.resSqmrhs;
      obj.resSqmrhs = 0;
    }
    return *this;
  }
  
  // For C++11, may need to add move constructors?
  // http://stackoverflow.com/questions/255612/dynamically-allocating-an-array-of-objects
};

#endif