#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;
using T = float;
void applyGmresIrQcu(void *fermion_out, void *fermion_in, void *gauge,
                     QcuParam *param, QcuParam *grid)
{
  // define for apply_wilson_gmres_ir
  LatticeSet<T> _set;
  _set.give(param->lattice_size, grid->lattice_size);
  _set.init();
  dptzyxcc2ccdptzyx<T>(gauge, &_set);
  ptzyxsc2psctzyx<T>(fermion_in, &_set);
  ptzyxsc2psctzyx<T>(fermion_out, &_set);
  LatticeGmresIr<T> _gmres_ir;
  _gmres_ir.give(&_set);
  _gmres_ir.init(fermion_out, fermion_in, gauge);
  _gmres_ir.run_test();
  _gmres_ir.end();
  ccdptzyx2dptzyxcc<T>(gauge, &_set);
  psctzyx2ptzyxsc<T>(fermion_in, &_set);
  psctzyx2ptzyxsc<T>(fermion_out, &_set);
  _set.end();
}