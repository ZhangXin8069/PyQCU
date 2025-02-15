// Copyright (c) 2018 Evan S Weinberg
// Base classes for linear operators.
// Includes a convenient wrapper for the function
// pointer convention used throughout this repository.

#ifndef QLINALG_OPERATOR_CLASS
#define QLINALG_OPERATOR_CLASS

#ifndef QLINALG_FCN_POINTER
#define QLINALG_FCN_POINTER
typedef void (*matrix_op_real)(double*,double*,void*);
typedef void (*matrix_op_cplx)(complex<double>*,complex<double>*,void*);
#endif

// Operator
template <typename T>
class Operator {
protected:
  int length;

public:
  Operator(int length)
   : length(length)
  {;}

  virtual void operator()(T* out, T* in) = 0;

  inline int get_length()
  {
    return length;
  }

  static void operator_apply(T* out, T* in, void* data)
  {
    Operator* op = static_cast<Operator*>(data);
    (*op)(out,in);
  }
};

// Function pointer wrapper
template <typename T>
class FunctionWrapper : public Operator<T> {
  friend Operator<T>;
  typedef void (*MatrixOp)(T*,T*,void*);

protected:
  MatrixOp fcn;
  void* data;

public:
  FunctionWrapper(MatrixOp fcn, void* data, int length)
   : Operator<T>(length), fcn(fcn), data(data)
  { ; }

  void operator()(T* out, T* in)
  {
    // Default: identity
    (*fcn)(out, in, data);
  }
};

#endif // QLINALG_OPERATOR_CLASS