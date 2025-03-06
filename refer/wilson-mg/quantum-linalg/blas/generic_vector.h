// Copyright (c) 2017 Evan S Weinberg
// Header file for templated vector operations.

#include <complex>
#include <random>

using std::complex; 
using std::polar;

#ifndef GENERIC_VECTOR
#define GENERIC_VECTOR

#ifndef PI
#define PI 3.14159265358979323846
#endif

// Trait for real version of a type
template <typename T>
struct ComplexBase {
  using type = T;
  static T real(T val) { return val; }
  static T imag(T val) { return 0; }
  static T conj(T val) { return val; }
};

template <typename T>
struct ComplexBase<std::complex<T> > {
  using type = T;
  static T real(std::complex<T> val) { return val.real(); }
  static T imag(std::complex<T> val) { return val.imag(); }
  static std::complex<T> conj(std::complex<T> val) { return std::complex<T>(val.real(), -val.imag()); }
};

template <typename T> struct Reducer { using type = T; };
template <> struct Reducer<float> { using type = double; };
template <> struct Reducer<std::complex<float>> { using type = std::complex<double>; };

template <typename T> struct RealReducer { using type = typename Reducer<typename ComplexBase<T>::type>::type; };

// Create and destroy a vector.
template<typename T> inline T* allocate_vector(int size)
{
  return new T[size];
}

template<typename T> inline void deallocate_vector(T** x)
{
  delete[] *x;
  *x = 0;
}

template<typename T> inline void deallocate_vector(T* const * x)
{
  delete[] *x;
}
// Zeroes a vector.
template<typename T> inline void zero_vector(T* x, int size)
{
  const T zero = static_cast<T>(0.0);
  for (int i = 0; i < size; i++)
  {
    x[i] = zero;
  }
}

// Special strided zero, follows blas conventions.
template<typename T> inline void zero_vector_blas(T* x, int xstep, int size)
{
  if (xstep == 1) { zero_vector(x, size); return; }

  const T zero = static_cast<T>(0.0);

  int ix = 0;
  for (int i = 0; i < size; i++)
  {
    x[ix] = zero;
    ix += xstep;
  }
}

// Assign a vector to a constant everywhere.
template <typename T, typename U = T> inline void constant_vector(T* x, U val, int size)
{
  for (int i = 0; i < size; i++)
  {
    x[i] = static_cast<T>(val);
  }
}

// Special strided constant, follows blas conventions.
template<typename T, typename U = T> inline void constant_vector_blas(T* x, int xstep, U val, int size)
{
  if (xstep == 1) { constant_vector(x, val, size); return; }

  int ix = 0;
  for (int i = 0; i < size; i++)
  {
    x[ix] = static_cast<T>(val);
    ix += xstep;
  }
}

// Special strided constant.
template<typename T, typename U = T> inline void constant_vector_stride(T* x, U val, int size, int start, int stride)
{
  for (int i = start; i < size; i += stride)
    x[i] = static_cast<T>(val);
}


// Random gaussian vector.
template<typename T> inline void gaussian(T* x, int size, std::mt19937 &generator, T deviation = 1.0, T mean = 0.0)
{
  // Generate a normal distribution.
  std::normal_distribution<> dist(0.0, deviation);
  for (int i = 0; i < size; i++)
  {
    x[i] = mean + static_cast<T>(dist(generator));
  }
}

// Random gaussian vector, random in real and imag.
template <typename T> inline void gaussian(complex<T>* x, int size, std::mt19937 &generator, T deviation = 1.0, complex<T> mean = 0.0)
{
  // Generate a normal distribution.
  std::normal_distribution<> dist(0.0, deviation);
  for (int i = 0; i < size; i++)
  {
    x[i] = std::complex<T>(real(mean) + static_cast<T>(dist(generator)), imag(mean) + static_cast<T>(dist(generator)));
  }
}

// Random gaussian vector, random in real, zero in imag.
template <typename T> inline void gaussian_real(complex<T>* x, int size, std::mt19937 &generator, T deviation = 1.0, complex<T> mean = 0.0)
{
  // Generate a normal distribution.
  std::normal_distribution<> dist(0.0, deviation);
  for (int i = 0; i < size; i++)
  {
    x[i] = std::complex<T>(real(mean) + static_cast<T>(dist(generator)), 0.0);
  }
}

// random vector on a uniform interval.
template <typename T> inline void random_uniform(T* x, int size, std::mt19937 &generator, T min, T max)
{
  std::uniform_real_distribution<> dist(min, max);
  for (int i = 0; i < size; i++)
  {
    x[i] = static_cast<T>(dist(generator));
  }
}

template <typename T> inline void random_uniform(complex<T>* x, int size, std::mt19937 &generator, T min, T max)
{
  std::uniform_real_distribution<> dist(min, max);
  for (int i = 0; i < size; i++)
  {
    x[i] = static_cast<complex<T>>(dist(generator));
  }
}

// random z2 vector
template <typename T> inline void random_z2(T* x, int size, std::mt19937 &generator)
{
  std::uniform_int_distribution<> dist(0, 1);
  for (int i = 0; i < size; i++)
  {
    x[i] = static_cast<T>(2*dist(generator)-1);
  }
}

// random z2 vector
template <typename T> inline void random_z2(complex<T>* x, int size, std::mt19937 &generator)
{
  std::uniform_int_distribution<> dist(0, 1);
  for (int i = 0; i < size; i++)
  {
    x[i] = std::complex<T>(static_cast<T>(2*dist(generator)-1), 0.0);
  }
}

// random z4 vector
template <typename T> inline void random_z4(complex<T>* x, int size, std::mt19937 &generator)
{
  std::uniform_int_distribution<> dist(0, 1);
  for (int i = 0; i < size; i++)
  {
    x[i] = std::complex<T>(static_cast<T>(2*dist(generator)-1), static_cast<T>(2*dist(generator)-1));
  }
}


// vectorized polar. applies x = polar(1.0, real(x));. Ignores imag(x).
template <typename T> inline void polar(complex<T>* x, int size)
{
  for (int i = 0; i < size; i++)
  {
    x[i] = static_cast<complex<T>>(polar(1.0, real(x[i])));
  }
}

// Take a real phase vector, turn it into complex phases.
template <typename T> inline void polar_vector(T* phase, complex<T>* u1, int size)
{
  for (int i = 0; i < size; i++)
  {
    u1[i] = polar(1.0, phase[i]);
  }
}

// vectorized abs.
template<typename T> inline void abs_vector(complex<T>*x, int size)
{
  for (int i = 0; i < size; i++)
  {
    x[i] = static_cast<complex<T>>(abs(x[i]));
  }
}

// vectorized arg.
template <typename T> inline void arg_vector(complex<T>* x, int size)
{
  for (int i = 0; i < size; i++)
  {
    x[i] = static_cast<complex<T>>(arg(x[i]));
  }
}

// Take a complex value, save as the real phase.
template <typename T> inline void arg_vector(complex<T>* u1, T* phase, int size)
{
  for (int i = 0; i < size; i++)
  {
    phase[i] = arg(u1[i]);
  }
}

// Exponentiate the components of a vector in place
template <typename T> inline void exp_vector(T* x, int size)
{
  for (int i = 0; i < size; i++)
  {
    x[i] = exp(x[i]);
  }
}

// Copy v2 into v1.
template<typename T, typename U = T> inline void copy_vector(T* v1, T* v2, int size)
{
  for (int i = 0; i < size; i++)
  {
    v1[i] = v2[i];
  }
}

// Special strided copy, follows blas conventions.
template<typename T, typename U = T> inline void copy_vector_blas(T* v1, T* v2, int xstep, int size)
{
  if (xstep == 1) { copy_vector(v1, v2, size); return; }

  int ix = 0;
  for (int i = 0; i < size; i++)
  {
    v1[ix] = v2[ix];
    ix += xstep;
  }
}

// Special strided copy, follows blas conventions.
template<typename T, typename U = T> inline void copy_vector_blas(T* v1, const int xstep, T* v2, const int ystep, const int size)
{
  if (xstep == 1 && ystep == 1) { copy_vector(v1, v2, size); return; }

  int ix = 0;
  int iy = 0;
  for (int i = 0; i < size; i++)
  {
    v1[ix] = v2[iy];
    ix += xstep;
    iy += ystep;
  }
}

// Add a constant to all components, y += a
template<typename T, typename U = T> inline void capx(U a, T* x, int size)
{
  for (int i = 0; i < size; i++)
  {
    x[i] += a;
  }
}

// Strided capx, y += a, where 'y' hops are specified.
template<typename T, typename U = T> inline void capx(U a, T* x, int xstep, int size)
{
  if (xstep == 1) { capx(a, x, size); return; }

  int ix = 0;
  for (int i = 0; i < size; i++)
  {
    x[ix] += a;
    ix += xstep;
  }
}

// Patterned capx, y += a, where a is a pattern which gets repeated.
template<typename T, typename U = T> inline void capx_pattern(U* a, const int asize, T* x, const int size)
{
  if (asize == 1) { capx(a[0], x, size); return; }

  int ix = 0;
  for (int i = 0; i < size; i++)
  {
    for (int ia = 0; ia < asize; ia++)
    {
      x[ix+ia] += a[ia];
    }
    ix += asize;
  }
}

// Invert each component of x.
template<typename T> inline void cinvx(T* x, const int size)
{
  for (int i = 0; i < size; i++)
  {
    x[i] = 1.0/x[i];
  }
}

// Rescale cax, x *= a
template<typename T, typename U = T> inline void cax(U a, T* x, int size)
{
  for (int i = 0; i < size; i++)
  {
    x[i] *= a;
  }
}

// Implemented strided cax, x *= a, where the 'x' hops are specified.
// Matches blas specification conventions.
template<typename T, typename U = T> inline void cax_blas(U a, T* x, int xstep, int size)
{
  if (xstep == 1) { cax(a, x, size); return; }

  int ix = 0;
  for (int i = 0; i < size; i++)
  {
    x[ix] *= a;
    ix += xstep;
  }
}

// Patterned cax, x *= a, where a is a pattern which gets repeated.
template<typename T, typename U = T> inline void cax_pattern(U* a, int asize, T* x, int size)
{
  if (asize == 1) { cax(a[0], x, size); return; }

  int ix = 0;
  for (int i = 0; i < size; i++)
  {
    for (int ia = 0; ia < asize; ia++)
    {
      x[ix+ia] *= a[ia];
    }
    ix += asize;
  }
}

// Implement caxy, y = a*x
template<typename T, typename U = T> inline void caxy(U a, T* x, T* y, int size)
{
  for (int i = 0; i < size; i++)
  {
    y[i] = a*x[i];
  }
}

// Implemented strided caxy, y = a*x, where the 'x' and 'y' hops are specified.
// Matches original blas specification.
template<typename T, typename U = T> inline void caxy_blas(U a, T* x, int xstep, T* y, int ystep, int size)
{
  if (xstep == 1 && ystep == 1) { caxy(a, x, y, size); return; }

  int ix = 0, iy = 0;
  for (int i = 0; i < size; i++)
  {
    y[iy] = a*x[ix];
    ix += xstep;
    iy += ystep;
  }
}

// Patterned cax, y = a*x, where a is a pattern which gets repeated.
template<typename T, typename U = T> inline void caxy_pattern(U* a, int asize, T* x, T* y, int size)
{
  if (asize == 1) { caxy(a[0], x, y, size); return; }

  int ix = 0;
  for (int i = 0; i < size; i++)
  {
    for (int ia = 0; ia < asize; ia++)
    {
      y[ix+ia] = a[ia] * x[ix+ia];
    }
    ix += asize;
  }
}

// Patterned and shuffled caxy, y = a*x, where a is a pattern which gets repeated.
// Permutes y[shuffle[i]] = a[i]*x[i]
template<typename T, typename U = T> inline void caxy_shuffle_pattern(U* a, int* shuffle, int asize, T* x, T* y, int size)
{
  if (asize == 1) { caxy(a[0], x, y, size); return; }

  int ix = 0;
  for (int i = 0; i < size; i++)
  {
    for (int ia = 0; ia < asize; ia++)
    {
      y[ix+shuffle[ia]] = a[ia] * x[ix+ia];
    }
    ix += asize;
  }
}

// Implement cxpy, y += x.
template<typename T> inline void cxpy(T* x, T* y, int size)
{
  for (int i = 0; i < size; i++)
  {
    y[i] += x[i];
  }
}

// Implement cxty, y *= x.
template<typename T> inline void cxty(T* x, T* y, int size)
{
  for (int i = 0; i < size; i++)
  {
    y[i] *= x[i];
  }
}

// Implement element-wise cxtyz, z = x*y.
template<typename T, typename U = T> inline void cxtyz(U* x, T* y, T* z, int size)
{
  for (int i = 0; i < size; i++)
  {
    z[i] = x[i]*y[i];
  }
}

// Implement caxpy, y += a*x
template<typename T, typename U = T> inline void caxpy(U a, T* x, T* y, int size)
{
  for (int i = 0; i < size; i++)
  {
    y[i] += a*x[i];
  }
}

// Implemented strided caxpy, y += a*x, where the 'x' and 'y' hops are specified.
// Matches original blas specification.
template<typename T, typename U = T> inline void caxpy_blas(U a, T* x, int xstep, T* y, int ystep, int size)
{
  if (xstep == 1 && ystep == 1) { caxpy(a, x, y, size); return; }

  int ix = 0, iy = 0;
  for (int i = 0; i < size; i++)
  {
    y[iy] += a*x[ix];
    ix += xstep;
    iy += ystep;
  }
}

// Special strided caxpy.
template<typename T, typename U = T> inline void caxpy_stride(U a, T* x, T* y, int size, int start, int stride)
{
  for (int i = start; i < size; i += stride)
    y[i] += a*x[i];
}

// Implement cxpay, y = x + a*y
template<typename T, typename U = T> inline void cxpay(T* x, U a, T* y, int size)
{
  for (int i = 0; i < size; i++)
  {
    y[i] = x[i] + a*y[i];
  }
}

// Implement caxpby, y = a*x + b*y
template<typename T, typename U = T> inline void caxpby(U a, T* x, U b, T* y, int size)
{
  for (int i = 0; i < size; i++)
  {
    y[i] = a*x[i] + b*y[i];
  }
}

// Implement cxpyz, z = x + y
template<typename T> inline void cxpyz(T* x, T* y, T* z, int size)
{
  for (int i = 0; i < size; i++)
  {
    z[i] = x[i] + y[i];
  }
}

// Implement cxpayz, z = x + a*y
template<typename T, typename U = T> inline void cxpayz(T* x, U a, T* y, T* z, int size)
{
  for (int i = 0; i < size; i++)
  {
    z[i] = x[i] + a*y[i];
  }
}

// Implement caxpbyz, z = a*x + b*y
template<typename T, typename U = T> inline void caxpbyz(U a, T* x, U b, T* y, T* z, int size)
{
  for (int i = 0; i < size; i++)
  {
    z[i] = a*x[i] + b*y[i];
  }
}

// Implement caxpbypz, z += a*x + b*y
template<typename T, typename U = T> inline void caxpbypz(U a, T* x, U b, T* y, T* z, int size)
{
  for (int i = 0; i < size; i++)
  {
    z[i] += a*x[i] + b*y[i];
  }
}

// Implement caxpbypcz, z = a*x + b*y + cz
template<typename T, typename U = T> inline void caxpbypcz(U a, T* x, U b, T* y, U c, T* z, int size)
{
  for (int i = 0; i < size; i++)
  {
  z[i] = a*x[i] + b*y[i] + c*z[i];
  }
}

// Implement caxpbypczw, w = a*x + b*y + cz
template<typename T, typename U = T> inline void caxpbypczw(U a, T* x, U b, T* y, U c, T* z, T* w, int size)
{
  for (int i = 0; i < size; i++)
  {
    w[i] = a*x[i] + b*y[i] + c*z[i];
  }
}


// Implement caxpyBzpx, y += a*x THEN x += b*z. 
template<typename T, typename U = T> inline void caxpyBzpx(U a, T* x, T* y, U b, T* z, int size)
{
  for (int i = 0; i < size; i++)
  {
    y[i] += a*x[i];
    x[i] += b*z[i];
  }
}

// Implement caxpyBxpz, y += a*x, z += b*x
template<typename T, typename U = T> inline void caxpyBxpz(U a, T* x, T* y, U b, T* z, int size)
{
  for (int i = 0; i < size; i++)
  {
  y[i] += a*x[i];
  z[i] += b*x[i];
  }
}

// Computes v1 dot v2.
template<typename T> inline typename Reducer<T>::type dot(T* v1, T* v2, int size)
{
  typename Reducer<T>::type res = static_cast<T>(0.0);
  for (int i = 0; i < size; i++)
  {
    res = res + ComplexBase<T>::conj(v1[i])*v2[i];
  }
  return res;
}

// Computes re(conj(v1) dot v2)
// trivial
template <typename T>
inline typename RealReducer<T>::type re_dot(T* v1, T* v2, int size)
{
  typename RealReducer<T>::type res = 0;
  for (int i = 0; i < size; i++)
  {
    res = res + ComplexBase<T>::real(ComplexBase<T>::conj(v1[i])*v2[i]);
  }
  return res;
}

// Sum over the contents.
template <typename T> inline T sum_vector(T* v1, int size)
{
  typename Reducer<T>::type res = 0;
  for (int i = 0; i < size; i++)
  {
    res = res + v1[i];
  }
  return res;
}

// Computes the vector norm squared.
template <typename T>
inline typename RealReducer<T>::type norm2sq(T* v1, int size)
{
  typename RealReducer<T>::type res = 0;
  for (int i = 0; i < size; i++)
  {
    res = res + ComplexBase<T>::real(ComplexBase<T>::conj(v1[i])*v1[i]);
  }
  return res;
}

// Return the infinity norm (max abs element)
template <typename T>
inline typename RealReducer<T>::type norminf(T* v1, int size)
{
  typename RealReducer<T>::type max_abs = 0;
  typename RealReducer<T>::type tmp = 0;
  for (int i = 0; i < size; i++)
  {
    tmp = abs(v1[i]);
    if (tmp > max_abs)
      max_abs = tmp;
  }
  return max_abs;
}

// Return |v1 - v2|^2
template <typename T>
inline typename RealReducer<T>::type diffnorm2sq(T* v1, T* v2, int size)
{
  typename RealReducer<T>::type res = 0;
  for (int i = 0; i < size; i++)
  {
    res = res + ComplexBase<T>::real(ComplexBase<T>::conj(v1[i] - v2[i])*(v1[i] - v2[i]));
  }
  return res;
}

template <typename T>
inline void normalize(T* v1, int size)
{
  typename RealReducer<T>::type res = 1.0/sqrt(norm2sq(v1, size));
  if (res > 0.0)
  {
    for (int i = 0; i < size; i++)
    {
      v1[i] *= res;
    }
  }
}

template <typename T>
inline void conj_vector(T* v1, int size)
{
  for (int i = 0; i < size; i++)
  {
    v1[i] = ComplexBase<T>::conj(v1[i]);
  }
}

// Make vector v1 orthogonal to vector v2. 
template <typename T>
inline void orthogonal(T* v1, T* v2, int size)
{
  T alpha = -dot<T>(v2, v1, size)/norm2sq(v2, size);   
  for (int i = 0; i < size; i++)
  {
    v1[i] = v1[i] + alpha*v2[i];
  }
}

// Apply some arbitrary function (supplied by a function pointer)
// with void* data to every site in the vector.
template <typename T>
inline void arb_local_function_vector(T* v1, void (*fcn)(int,T&,void*), void* extra_data, int size)
{
  for (int i = 0; i < size; i++)
  {
    (*fcn)(i,v1[i], extra_data);
  }
}

#endif // GENERIC_VECTOR

