import numpy as np
import cupy
import pyquda
import sys
from opt_einsum import contract

lqcd_analysis = "/public/home/huzhch_p/application/lqcd_analysis/"
sys.path.append(lqcd_analysis) 
import include
from include import MPI_init,MPI_gather_np,MPI_scatter_np,MPI_scatter_np,MPI_get_nparr,MPI_slice_extract,MPI_print
from include import Inform_clover
from include import general as general
from include.hadspec import get_pmodes
# from include.include_phys import gamma
from timeit import default_timer as timer

easy_dist = "/public/home/huzhch_p/application/EasyDistillation/"
sys.path.append(easy_dist) 
from lattice import set_backend, get_backend, check_QUDA
from pyquda_utils import io,core
import pickle
from time import perf_counter

def main(
        ens,cfg_id, 
        grid_size, mom, 
        mq,precise,
    ):
    comm,mpisize, rank = MPI_init()
    root = (rank==0)
    set_backend("cupy")
    backend = get_backend()
    clover = Inform_clover(ens)
    latt_size = clover.size
    
    Lx, Ly, Lz, Lt = latt_size
    Gx, Gy, Gz, Gt = grid_size
    Lx, Ly, Lz, Lt = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]

    MPI_print(clover.cfg_name)
    MPI_print(clover.size)
    
    xi_0, nu = 1.0, 1.0
    coeff_r, coeff_t = float(clover.vI0)**(-3),float(clover.vI0)**(-3)
    
    set_backend("cupy")
    backend = get_backend()
    grid_size=[1,1,1,4]
    
    g5   = backend.asarray(include.include_phys.gamma(15))
    latt_info = pyquda.LatticeInfo(latt_size=latt_size, t_boundary=-1, anisotropy=xi_0/nu)
    pyquda.init(grid_size=grid_size, backend="cupy", resource_path=None)
    
    gauge_folder = "/public/home/huzhch_p/Configure_clover/"
    gauge_file   = f"{gauge_folder}/{clover.cfg_name}/{clover.cfg_name}_cfg_{cfg_id}.lime"
    gauge        = io.readQIOGauge(gauge_file)
    gauge.smearSTOUT(1, 0.125, 4)
    
    dslash = core.getDirac(latt_info,float(mq),float(precise), 1000, xi_0, coeff_t, coeff_r,[[4,6,2,6], [6,4,6,3]])
    dslash.loadGauge(gauge)



    Plist = get_pmodes(int(mom)**2) 
    MPI_print(Plist)
    pnum = len(Plist)
    # c2 = np.zeros( (pnum,4,4,clover.size[3],clover.size[3]), dtype=complex )
    c2 = np.zeros( (pnum,16,4,4,clover.size[3],clover.size[3]), dtype=complex )

    for tsrc in range(Lt*Gt):
        
        ##invertion
        s = perf_counter()
        MPI_print("tsrc=",tsrc)
        # prop = core.invert(dirac=dslash, source_type="wall", t_srce=[0, 0, 0, tsrc]).data.get()
        prop = core.invert(dirac=dslash, source_type="point", t_srce=[0, 0, 0, tsrc]).data.get()
        invert_time = perf_counter() - s
        general.get_gpu_memory()
        MPI_print(prop.shape)


        ##contraction
        s = perf_counter()
        prop = backend.asarray( core.lexico(  prop,  [0, 1, 2, 3, 4] ) )
        MPI_print(prop.shape)
        g5   = backend.asarray(include.include_phys.gamma(15))
        anti_prop = contract(  "ab, tzyx cb ij , cd -> tzyx ad ji  ",g5,prop.conj(),g5  )

        for i in range(16):
            gm   = backend.asarray(include.include_phys.gamma(i))
            if (mom == 0):
                Px,Py,Pz = 0,0,0 
                _c2 = contract( "tzyx ab ij, bc, tzyx cd ji, de -> aet", anti_prop, gm, prop, gm ).get()
                _c2 = core.gatherLattice(_c2, [2, -1, -1, -1]) 
                if root:
                    print(_c2.shape)
                    print( contract("aat-> t",_c2)  )
                c2[0,i,...,tsrc,:] = _c2
            else:
                iP = -1
                for P in Plist:
                    iP += 1 
                    Pz,Py,Px = P
                    pvector = np.array([Pz,Py,Px])
                    MPI_print( f"momentum phase: {pvector}" )

                    z,y,x   = np.meshgrid(np.arange(Lz), np.arange(Ly), np.arange(Lx), indexing='ij')
                    cords   = np.stack((z,y,x), axis=-1)
                    phase   = np.exp( -2*1j*np.pi/(Lx*Gx)*contract("zyxv,v->zyx", cords, pvector) ) 

                    _c2 = contract( 
                        "tzyx ab ij, bc, tzyx cd ji,   de,         zyx            -> aet", 
                        anti_prop,   gm,     prop,     gm,   backend.asarray(phase) 
                    ).get()
                    _c2 = core.gatherLattice(_c2, [2, -1, -1, -1]) 
                    if root:
                        print(_c2.shape)
                        print( contract("aat-> t",_c2)  )
                    c2[iP,i,...,tsrc,:] = _c2
        contraction_time = perf_counter() - s
        free, total = cupy.cuda.runtime.memGetInfo()

        MPI_print(
            f"Tsrc = {tsrc}:  inv t = {invert_time:.4f} sec, contraction t = {contraction_time:.4f} sec, device mem: {(total - free) / 1024**3} GB, free:{free / 1024**3} GB."
        )
        general.get_memory()

        # break

    if root:
        iP = -1
        for P in Plist:
            iP += 1 
            Pz,Py,Px = P
            outc2path = general.mk_outpath("./output",clover.cfg_bfn,"Meson_loopGM")
            outlink   = general.mk_outpath(outc2path,cfg_id)
            # c2_file   = outlink + f"PionC2_NoTrace_PiontSource_mq{mq}_precise{precise}_Px{Px}Py{Py}Pz{Pz}.pkl"
            c2_file   = outlink + f"MesonC2_NoTraceLoopGM_PiontSource_mq{mq}_precise{precise}_Px{Px}Py{Py}Pz{Pz}.pkl"
            pickle.dump(c2[iP],open(c2_file,"wb"))
            print(general.get_func_name(), f":save c2 to [{c2_file}]" )
    comm.Barrier() 

    dslash.destroy()
    return 0

if __name__ == '__main__':
    comm,mpisize,rank = MPI_init()
    import traceback
    from include import bash2py
    try:
        import argparse
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('-ens',          type=int,             help='ensemble')
        parser.add_argument('-cfg_id',       type=int,             help='conf_id')
        parser.add_argument('-grid_size',    type=bash2py.IntList, help='[Gx,Gy,Gz,Gt]')
        parser.add_argument('-mom',          type=int,             help='momentum')
        parser.add_argument('-mq',           type=str,           help='quark mass')
        parser.add_argument('-precise',      type=str,           help='...')
        args = parser.parse_args()
        if (rank==0): print(args)
        main(
            args.ens,args.cfg_id, 
            args.grid_size, args.mom, 
            args.mq,args.precise
        )

    except Exception as e:
        general.cprint(f"Error in process {rank}: {str(e)}")
        traceback.print_exc()
        comm.Abort(1)

