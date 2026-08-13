#!/usr/bin/env python3
"""Verify the C++ SCHUR MG output correctness on 8x8x8x16."""
import torch, os, sys, time
from pyqcu import tools, dslash
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "csm", os.path.join(os.path.dirname(os.path.abspath(__file__)), "conftest.schur.multigrid.py"))
_csm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_csm)
build_config = _csm.build_config
build_schur_levels = _csm.build_schur_levels

def main():
    Lx,Ly,Lz,Lt=8,8,8,16; MASS=0.05; ATOL=1e-6
    av = build_config(Lx,Ly,Lz,Lt,MASS,ATOL,2,[12,48],[2,2,2,2],10,200,1e3,define._LAT_C64_)
    KAPPA=1.0/(2*MASS+8); device=torch.device('cuda'); dt=define.dtype(define._LAT_C64_)
    ls=define.lat_shape(params)
    g=torch.zeros([2,3,3,4]+ls,dtype=dt,device=device)
    fi=torch.randn([2,4,3]+ls,dtype=dt,device=device)
    fo_ref=torch.zeros_like(fi); fo_mg=torch.zeros_like(fi)
    ce=torch.zeros([4,3,4,3]+ls,dtype=dt,device=device)
    cei=torch.zeros_like(ce); coo=torch.zeros_like(ce); coi=torch.zeros_like(ce)
    params[define._SET_INDEX_]=0; params[define._SET_PLAN_]=-1
    qcu.applyInitQcu(set_ptrs,params,av); qcu.applyGaussGaugeQcu(g,set_ptrs,params)
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=2; params[define._PARITY_]=0
    qcu.applyInitQcu(set_ptrs,params,av); qcu.applyCloversQcu(ce,cei,g,set_ptrs,params)
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=2; params[define._PARITY_]=1
    qcu.applyInitQcu(set_ptrs,params,av); qcu.applyCloversQcu(coo,coi,g,set_ptrs,params)

    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=1; params[define._VERBOSE_]=0
    qcu.applyInitQcu(set_ptrs,params,av)
    qcu.applyCloverBistabCgQcu(fo_ref,fi,g,ce,coo,cei,coi,set_ptrs,params)
    qcu_U=tools.poooxyzt2oooxyzt(g); qcu_src=tools.poooxyzt2oooxyzt(fi); qcu_ref=tools.poooxyzt2oooxyzt(fo_ref)
    ref_cl=dslash.make_clover(qcu_U,kappa=KAPPA)
    ref_res=tools.norm(dslash.give_wilson(qcu_ref,qcu_U,KAPPA,True)+dslash.give_clover(qcu_ref,ref_cl)-qcu_src)/tools.norm(qcu_src)
    print(f"[ref] BiStabCG res={ref_res:.3e}")

    op=dslash.operator(U=qcu_U,clover_term=ref_cl,kappa=torch.Tensor([KAPPA]),support_parity=True,verbose=False)
    S=op.matvec_parity
    lonvs,hnn_l,hdg_l,sit_l=build_schur_levels(op,S,2,[12,48],[2,2,2,2],[Lx,Ly,Lz,Lt],48,dt,device,2)
    for fl in range(len(lonvs)):
        set_ptrs[30+4*fl+0]=lonvs[fl].contiguous().data_ptr()
        set_ptrs[30+4*fl+1]=hnn_l[fl].contiguous().data_ptr()
        set_ptrs[30+4*fl+2]=hdg_l[fl].contiguous().data_ptr()
        set_ptrs[30+4*fl+3]=sit_l[fl].contiguous().data_ptr()

    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=1; params[define._VERBOSE_]=0
    qcu.applyInitQcu(set_ptrs,params,av)
    torch.cuda.synchronize(); t0=time.perf_counter()
    qcu.applyCloverMultigridQcu(fo_mg,fi,g,ce,coo,cei,coi,set_ptrs,params)
    torch.cuda.synchronize(); mg_time=time.perf_counter()-t0
    print(f"[mg] time={mg_time*1000:.1f}ms")

    qcu_mg=tools.poooxyzt2oooxyzt(fo_mg)
    mg_res=tools.norm(dslash.give_wilson(qcu_mg,qcu_U,KAPPA,True)+dslash.give_clover(qcu_mg,ref_cl)-qcu_src)/tools.norm(qcu_src)
    mg_vs_ref=tools.norm(qcu_mg-qcu_ref)/tools.norm(qcu_ref)
    print(f"[mg] res={mg_res:.4e} vs_ref={mg_vs_ref:.4e}")

    # Check the SCHUR residual of the odd part directly
    x_o_mg = tools.oooxyzt2poooxyzt(fo_mg)[1]
    b_eo=tools.oooxyzt2poooxyzt(qcu_src.reshape([12]+list(qcu_src.shape)[2:]))
    b__o=op.give_b_parity(b_e=b_eo[0],b_o=b_eo[1])
    schur_res=tools.norm(b__o - S(x_o_mg))/tools.norm(b__o)
    print(f"[mg] SCHUR residual = {schur_res:.4e}")

    # Reconstruct x_e and check full residual manually
    x_e_mg=op.give_x_e(b_e=b_eo[0],x_o=x_o_mg)
    x_full=tools.poooxyzt2oooxyzt(torch.stack([x_e_mg,x_o_mg],dim=0)).reshape(qcu_mg.shape)
    full_res=tools.norm(dslash.give_wilson(x_full,qcu_U,KAPPA,True)+dslash.give_clover(x_full,ref_cl)-qcu_src)/tools.norm(qcu_src)
    print(f"[mg] reconstructed full residual = {full_res:.4e}")

if __name__=="__main__":
    main()
