# coding: utf-8
from mpi4py import MPI
if not MPI.Is_initialized():
    MPI.Init()

try:
    import faulthandler
    comm = MPI.Comm()
    faulthandler.enable(open(f"fault_rank{comm.Get_rank()}.log", "w"))
except Exception as e:
    print(f"Error: {e}")


def info():
    try:
        print(
            '''
    @@@@@@######QCU NOTES START######@@@@@@@
    Guide:
    0. Required: MPI(e.g. 4.1.2), CUDA(e.g. 12.4), CMAKE(e.g. 3.22.1), GCC(e.g. 11.4.0), HDF5-MPI(e.g. 1.10.7,'apt install libhdf5-mpi-dev && export HDF5_MPI="ON" && pip install --no-binary=h5py h5py').
    1. The libqcu.so was compiled when pyqcu setup in download_path/PyQCU/lib, please add this path to your LD_LIBRARY_PATH.
    2. The QCU(PyQCU) splite grid by x->y->z->t, lattice by x->y->z->t->p->d[x,y,z,t]->c->c or x->y->z->t->c->s(->p) and x->y->z->t->c->s->c->s(->p).
    3. The QUDA(PyQUDA) splite grid by t->z->y->x, lattice by c->c->x->y->z->t->p->d[x,y,z,t] or c->s->x->y->z->t(->p) and c->s->c->s->x->y->z->t(->p).
    4. The QCU input params in numpy array(dtype=np.int32), argv in  numpy array(dtype=np.float32 or float64) array, set_ptrs in numpy array(dtype=np.int64), other in cupy array(dtype=cp.complex64 or complex128).
    5. The smallest lattice size is (wilson:x=4,y=4,z=4,t=4;clover:x=8,y=8,z=8,t=8) that QCU support (when '#define _BLOCK_SIZE_ 32 // for test small lattice').
    References:
    [0] 刘川. 格点量子色动力学导论. 北京大学出版社, 2017.07.
    [1] 蒋翔宇. 轻强子性质的格点QCD研究. 中国科学院高能物理研究所, 2023.06.
    [2] Babich R, Clark M A, Joó B. Parallelizing the QUDA library for multi-GPU calculations in lattice quantum chromodynamics[C]//SC'10: Proceedings of the 2010 ACM/IEEE International Conference for High Performance Computing, Networking, Storage and Analysis. IEEE, 2010.01.11.
    [3] M. Rottmann. Adaptive domain decomposition multigrid for lattice QCD, Ph.D. thesis, Wuppertal, Univ., Diss., 2016.
    [4] Brower R C, Clark M A, Weinberg E, et al. Multigrid for chiral lattice fermions: Domain wall[J]. Physical Review D, 2020.
    @@@@@@######QCU NOTES END######@@@@@@@
    ''')
    except Exception as e:
        print(f"Error: {e}")
