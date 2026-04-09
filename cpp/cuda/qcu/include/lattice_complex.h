#ifndef _LATTICE_COMPLEX_H
#define _LATTICE_COMPLEX_H
#include "./include.h"
namespace qcu
{
  template <typename T>
  struct LatticeComplex
  {
    using _data_type = typename std::conditional<
        std::is_same<T, double>::value, double2,
        typename std::conditional<
            std::is_same<T, float>::value, float2,
            void>::type>::type;
    _data_type _data;
    __host__ __device__ __inline__ LatticeComplex(const T &_real = 0.0,
                                                  const T &_imag = 0.0)
    {
      _data.x = _real;
      _data.y = _imag;
    }
    __host__ __device__ __inline__ LatticeComplex &
    operator=(const LatticeComplex &other)
    {
      _data.x = other._data.x;
      _data.y = other._data.y;
      return *this;
    }
    __host__ __device__ __inline__ LatticeComplex &
    operator=(const T &other)
    {
      _data.x = other;
      _data.y = 0;
      return *this;
    }
    __host__ __device__ __inline__ LatticeComplex
    operator+(const LatticeComplex &other) const
    {
      return LatticeComplex(_data.x + other._data.x, _data.y + other._data.y);
    }
    __host__ __device__ __inline__ LatticeComplex
    operator-(const LatticeComplex &other) const
    {
      return LatticeComplex(_data.x - other._data.x, _data.y - other._data.y);
    }
    __host__ __device__ __inline__ LatticeComplex
    operator*(const LatticeComplex &other) const
    {
      return LatticeComplex(_data.x * other._data.x - _data.y * other._data.y,
                            _data.x * other._data.y + _data.y * other._data.x);
    }
    __host__ __device__ __inline__ LatticeComplex
    operator*(const T &other) const
    {
      return LatticeComplex(_data.x * other, _data.y * other);
    }
    __host__ __device__ __inline__ LatticeComplex
    operator/(const LatticeComplex &other) const
    {
      T denom =
          other._data.x * other._data.x + other._data.y * other._data.y;
      return LatticeComplex(
          (_data.x * other._data.x + _data.y * other._data.y) / denom,
          (_data.y * other._data.x - _data.x * other._data.y) / denom);
    }
    __host__ __device__ __inline__ LatticeComplex
    operator/(const T &other) const
    {
      return LatticeComplex(_data.x / other, _data.y / other);
    }
    __host__ __device__ __inline__ LatticeComplex operator-() const
    {
      return LatticeComplex(-_data.x, -_data.y);
    }
    __host__ __device__ __inline__ bool
    operator==(const LatticeComplex &other) const
    {
      return (_data.x == other._data.x && _data.y == other._data.y);
    }
    __host__ __device__ __inline__ bool
    operator!=(const LatticeComplex &other) const
    {
      return !(*this == other);
    }
    __host__ __device__ __inline__ LatticeComplex &
    operator+=(const LatticeComplex &other)
    {
      _data.x = _data.x + other._data.x;
      _data.y = _data.y + other._data.y;
      return *this;
    }
    __host__ __device__ __inline__ LatticeComplex &
    operator-=(const LatticeComplex &other)
    {
      _data.x = _data.x - other._data.x;
      _data.y = _data.y - other._data.y;
      return *this;
    }
    __host__ __device__ __inline__ LatticeComplex &
    operator*=(const LatticeComplex &other)
    {
      _data.x = _data.x * other._data.x - _data.y * other._data.y;
      _data.y = _data.x * other._data.y + _data.y * other._data.x;
      return *this;
    }
    __host__ __device__ __inline__ LatticeComplex &
    operator*=(const T &other)
    {
      _data.x = _data.x * other;
      _data.y = _data.y * other;
      return *this;
    }
    __host__ __device__ __inline__ LatticeComplex &
    operator/=(const LatticeComplex &other)
    {
      T denom =
          other._data.x * other._data.x + other._data.y * other._data.y;
      _data.x = (_data.x * other._data.x + _data.y * other._data.y) / denom;
      _data.y = (_data.y * other._data.x - _data.x * other._data.y) / denom;
      return *this;
    }
    __host__ __device__ __inline__ LatticeComplex &
    operator/=(const T &other)
    {
      _data.x = _data.x / other;
      _data.y = _data.y / other;
      return *this;
    }
    __host__ __device__ __inline__ T real() const
    {
      return _data.x;
    }
    __host__ __device__ __inline__ T imag() const
    {
      return _data.y;
    }
    __host__ __device__ __inline__ LatticeComplex conj() const
    {
      return LatticeComplex(_data.x, -_data.y);
    }
    __host__ __device__ __inline__ LatticeComplex multi_minus() const
    {
      return LatticeComplex(-_data.x, -_data.y);
    }
    __host__ __device__ __inline__ LatticeComplex multi_i() const
    {
      return LatticeComplex(-_data.y, _data.x);
    }
    __host__ __device__ __inline__ LatticeComplex multi_i(T val) const
    {
      return LatticeComplex(-_data.y * val, _data.x * val);
    }
    __host__ __device__ __inline__ LatticeComplex multi_none() const
    {
      return LatticeComplex(_data.x, _data.y);
    }
    __host__ __device__ __inline__ LatticeComplex multi_none(T val) const
    {
      return LatticeComplex(_data.x * val, _data.y * val);
    }
    __host__ __device__ __inline__ LatticeComplex multi_minus_i() const
    {
      return LatticeComplex(_data.y, -_data.x);
    }
    __host__ __device__ __inline__ T norm2() const
    {
      return sqrt(_data.x * _data.x + _data.y * _data.y);
    }
    friend std::ostream &operator<<(std::ostream &output, const LatticeComplex &_)
    {
      output << "(" << _._data.x << "," << _._data.y << "i"
             << ")";
      return output;
    }
  };
}
#endif