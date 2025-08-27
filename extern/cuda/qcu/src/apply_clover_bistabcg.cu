// #include "../python/pyqcu.h"
// #include "../include/qcu.h"
// #pragma optimize(5)
// using namespace qcu;
// void applyWilsonBistabCgQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _set_ptrs, long long _params)
// {
//   cudaDeviceSynchronize();
//   void *fermion_out = (void *)_fermion_out;
//   void *fermion_in = (void *)_fermion_in;
//   void *gauge = (void *)_gauge;
//   void *set_ptrs = (void *)_set_ptrs;
//   void *params = (void *)_params;
//   int set_index = static_cast<int *>(params)[_SET_INDEX_];
//   int data_type = static_cast<int *>(params)[_DATA_TYPE_];
//   if (data_type == _LAT_C64_)
//   {
//     LatticeSet<float> *set_ptr = static_cast<LatticeSet<float> *>((void *)(static_cast<long long *>(set_ptrs)[set_index])); // define for apply_wilson_bistabcg
//     // dptzyxcc2ccdptzyx<float>(gauge, &_set);
//     // ptzyxsc2psctzyx<float>(fermion_in, &_set);
//     // ptzyxsc2psctzyx<float>(fermion_out, &_set);
//     LatticeWilsonBistabCg<float> _bistabcg;
//     _bistabcg.give(set_ptr);
//     _bistabcg.init(fermion_out, fermion_in, gauge);
//     if (set_ptr->host_params[_VERBOSE_])
//     {
//       _bistabcg.run_test();
//     }
//     else
//     {
//       _bistabcg.run();
//     }
//     _bistabcg.end();
//     // ccdptzyx2dptzyxcc<float>(gauge, &_set);
//     // psctzyx2ptzyxsc<float>(fermion_in, &_set);
//     // psctzyx2ptzyxsc<float>(fermion_out, &_set);
//   }
//   else if (data_type == _LAT_C128_)
//   {
//     LatticeSet<double> *set_ptr = static_cast<LatticeSet<double> *>((void *)(static_cast<long long *>(set_ptrs)[set_index])); // define for apply_wilson_bistabcg
//     // dptzyxcc2ccdptzyx<double>(gauge, &_set);
//     // ptzyxsc2psctzyx<double>(fermion_in, &_set);
//     // ptzyxsc2psctzyx<double>(fermion_out, &_set);
//     LatticeWilsonBistabCg<double> _bistabcg;
//     _bistabcg.give(set_ptr);
//     _bistabcg.init(fermion_out, fermion_in, gauge);
//     if (set_ptr->host_params[_VERBOSE_])
//     {
//       _bistabcg.run_test();
//     }
//     else
//     {
//       _bistabcg.run();
//     }
//     _bistabcg.end();
//     // ccdptzyx2dptzyxcc<double>(gauge, &_set);
//     // psctzyx2ptzyxsc<double>(fermion_in, &_set);
//     // psctzyx2ptzyxsc<double>(fermion_out, &_set);
//   }
//   else
//   {
//     printf("data_type error\n");
//   }
//   cudaDeviceSynchronize();
// }