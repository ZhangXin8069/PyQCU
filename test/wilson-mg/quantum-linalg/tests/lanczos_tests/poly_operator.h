// Copyright (c) 2018 Evan S Weinberg
// Simple polynomials of linear operators.

#ifndef QLINALG_POLY_OPERATOR_CLASS
#define QLINALG_POLY_OPERATOR_CLASS

#include "operator.h"

// Linear map from [a,b] to [-1,1]
template <typename T, typename U>
class LinearMapToUnit : public Operator<T> {
protected:
  using Operator<T>::length;

  Operator<T>* op;
  const U a,b;
  T* tmp;

  U slope, shift;

public:
  LinearMapToUnit(Operator<T>* op, const U a, const U b)
   : Operator<T>(op->get_length()), op(op), a(a), b(b)
  {
    tmp = allocate_vector<T>(length);
    U c = 0.5*(a+b);
    U d = 0.5*(b-a);

    slope = 1.0/d;
    shift = -c*slope;
  }

  ~LinearMapToUnit()
  {
    deallocate_vector(&tmp);
  }

  void operator()(T* out, T* in)
  {
    zero_vector(tmp, length);
    (*op)(tmp, in);
    caxpbyz(slope, tmp, shift, in, out, length);
  }
};

// Square the operator
template <typename T>
class Squared : public Operator<T> {
protected:
  using Operator<T>::length;

  Operator<T>* op;
  T* tmp;

public:
  Squared(Operator<T>* op)
   : Operator<T>(op->get_length()), op(op)
  {
    tmp = allocate_vector<T>(length);
  }

  ~Squared()
  {
    deallocate_vector(&tmp);
  }

  void operator()(T* out, T* in)
  {
    zero_vector(tmp, length);
    (*op)(tmp,in);
    (*op)(out,tmp);
  }
};

// Apply the series of 1/(1+x)
template <typename T>
class OneOverOnePlusX : public Operator<T> {
protected:
  using Operator<T>::length;

  Operator<T>* op;
  int order;
  T* tmp;
  T* tmp2;

public:
  OneOverOnePlusX(Operator<T>* op, int order)
   : Operator<T>(op->get_length()), op(op), order(order)
  {
    tmp = allocate_vector<T>(length);
    tmp2 = allocate_vector<T>(length);
  }

  ~OneOverOnePlusX()
  {
    deallocate_vector(&tmp);
    deallocate_vector(&tmp2);
  }

  void operator()(T* out, T* in)
  {
    // Apply the series for 1/(1+x) to whatever order
    copy_vector(out, in, length);
    copy_vector(tmp, in, length);

    for (int i = 0; i < order; i++)
    {
      // Apply the operator
      (*op)(tmp2, tmp);

      // accumulate
      caxpy((i%2 == 0) ? -1.0 : 1.0, tmp2, out, length);

      // pointer swap
      std::swap(tmp, tmp2);

    }

  }
};

#endif // QLINALG_POLY_OPERATOR_CLASS