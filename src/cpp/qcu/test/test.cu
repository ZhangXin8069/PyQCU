#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;
using T = float;
int main()
{
  MPI_Init(NULL, NULL);
  void *gauge;
  void *fermion_in;
  void *fermion_out;
  QcuParam param;
  QcuParam grid;
  int parity, lat_4dim_SC, lat_4dim_DCC;
  { // io
    std::stringstream filename;
    filename << "wilson-bistabcg-gauge_-32-32-32-32-1048576-1-1-1-1-0-0-1-0-f.bin";
    get_filename(filename, param, parity, grid);
    // lat_4dim_SC = param.lattice_size[_X_] * param.lattice_size[_Y_] * param.lattice_size[_Z_] * param.lattice_size[_T_] * _LAT_SC_ / _EVEN_ODD_; // dslash
    lat_4dim_SC = param.lattice_size[_X_] * param.lattice_size[_Y_] * param.lattice_size[_Z_] * param.lattice_size[_T_] * _LAT_SC_;              // cg
    lat_4dim_DCC = param.lattice_size[_X_] * param.lattice_size[_Y_] * param.lattice_size[_Z_] * param.lattice_size[_T_] * _LAT_DCC_ / _EVEN_ODD_;
    { // fermion_out
      std::stringstream filename;
      // filename << "wilson-bistabcg-fermion-out_-32-32-32-32-1048576-1-1-1-1-0-0-1-0-f.bin";
      cudaDeviceSynchronize();
      cudaMalloc(&fermion_out, lat_4dim_SC * _REAL_IMAG_ * sizeof(T));
      cudaDeviceSynchronize();
      // device_load<T>(fermion_out, lat_4dim_SC * _REAL_IMAG_, filename.str());
    }
    { // fermion_in
      std::stringstream filename;
      filename << "wilson-bistabcg-fermion-in_-32-32-32-32-1048576-1-1-1-1-0-0-1-0-f.bin";
      cudaDeviceSynchronize();
      cudaMalloc(&fermion_in, lat_4dim_SC * _REAL_IMAG_ * sizeof(T));
      cudaDeviceSynchronize();
      device_load<T>(fermion_in, lat_4dim_SC * _REAL_IMAG_, filename.str());
    }
    { // gauge
      std::stringstream filename;
      filename << "wilson-bistabcg-gauge_-32-32-32-32-1048576-1-1-1-1-0-0-1-0-f.bin";
      cudaDeviceSynchronize();
      cudaMalloc(&gauge, lat_4dim_DCC * _EVEN_ODD_ * _REAL_IMAG_ * sizeof(T));
      cudaDeviceSynchronize();
      device_load<T>(gauge, lat_4dim_DCC * _EVEN_ODD_ * _REAL_IMAG_, filename.str());
    }
  }
  // applyWilsonDslashQcu(fermion_out, fermion_in, gauge,
  //                &param, parity, &grid);
  // applyCloverDslashQcu(fermion_out, fermion_in, gauge,
  //                      &param, parity, &grid);
  applyBistabCgQcu(fermion_out, fermion_in, gauge,
                   &param, &grid);
  // applyCgQcu(fermion_out, fermion_in, gauge,
  //                  &param, &grid);
  { // io
    std::stringstream filename;
    filename << "_wilson-bistabcg-fermion-out_-32-32-32-32-1048576-1-1-1-1-0-0-1-0-f.bin";
    device_save<T>(fermion_out, lat_4dim_SC * _REAL_IMAG_, filename.str());
  }
  cudaFree(gauge);
  cudaFree(fermion_in);
  cudaFree(fermion_out);
  MPI_Finalize();
  return 0;
}