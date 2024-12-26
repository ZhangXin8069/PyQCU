#ifndef _QCU_H
#define _QCU_H
#pragma once
#include "./define.h"
#include "./include.h"
#include "./lattice_cg.h"
#include "./lattice_bistabcg.h"
#include "./lattice_gmres_ir.h"
#include "./lattice_clover_dslash.h"
#include "./lattice_complex.h"
#include "./lattice_mpi.h"
#include "./lattice_cuda.h"
#include "./lattice_set.h"
#include "./lattice_wilson_dslash.h"
#ifdef __cplusplus
extern "C"
{
#endif
  typedef struct QcuParam_s
  {
    int lattice_size[4];
  } QcuParam;
  void testWilsonDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                     QcuParam *param, int parity);
  void applyWilsonDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                      QcuParam *param, int parity, QcuParam *grid);
  void testCloverDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                           QcuParam *param, int parity);
  void applyCloverDslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                            QcuParam *param, int parity, QcuParam *grid);
  void applyBistabCgQcu(void *fermion_out, void *fermion_in, void *gauge,
                        QcuParam *param, QcuParam *grid);
  void applyCgQcu(void *fermion_out, void *fermion_in, void *gauge,
                  QcuParam *param, QcuParam *grid);
  void applyGmresIrQcu(void *fermion_out, void *fermion_in, void *gauge,
                       QcuParam *param, QcuParam *grid);
#ifdef __cplusplus
}
#endif
#endif