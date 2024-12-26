#include "../include/pyqcu.h"
#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;
using T = float;
void applyWilsonDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                    void *params, void *argv)
{
  // define for apply_wilson_dslash
  LatticeSet<T> _set;
  _set.give(params, argv);
  _set.init();
  dptzyxcc2ccdptzyx<T>(gauge, &_set);
  tzyxsc2sctzyx<T>(fermion_in, &_set);
  tzyxsc2sctzyx<T>(fermion_out, &_set);
  LatticeWilsonDslash<T> _wilson_dslash;
  _wilson_dslash.give(&_set);
  _wilson_dslash.run_test(fermion_out, fermion_in, gauge);
  ccdptzyx2dptzyxcc<T>(gauge, &_set);
  sctzyx2tzyxsc<T>(fermion_in, &_set);
  sctzyx2tzyxsc<T>(fermion_out, &_set);
  _set.end();
}