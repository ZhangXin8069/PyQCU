#include "../python/pyqcu.h"
#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;
using T = float;
void applyWilsonBistabCgQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _params, long long _argv)
{
  void *fermion_out = (void *)_fermion_out;
  void *fermion_in = (void *)_fermion_in;
  void *gauge = (void *)_gauge;
  void *argv = (void *)_argv;
  void *params = (void *)_params;
  // define for apply_wilson_bistabcg
  LatticeSet<T> _set;
  _set.give(params, argv);
  _set.init();
  // dptzyxcc2ccdptzyx<T>(gauge, &_set);
  // ptzyxsc2psctzyx<T>(fermion_in, &_set);
  // ptzyxsc2psctzyx<T>(fermion_out, &_set);
  LatticeBistabCg<T> _bistabcg;
  _bistabcg.give(&_set);
  _bistabcg.init(fermion_out, fermion_in, gauge);
  _bistabcg.run_test();
  _bistabcg.end();
  // ccdptzyx2dptzyxcc<T>(gauge, &_set);
  // psctzyx2ptzyxsc<T>(fermion_in, &_set);
  // psctzyx2ptzyxsc<T>(fermion_out, &_set);
  _set.end();
}