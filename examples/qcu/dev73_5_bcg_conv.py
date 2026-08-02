#!/usr/bin/env python3
"""dev73_5 BiStabCG 收敛轨迹采集。

用 instrumented libqcu.so（/tmp/qcu-printbcg，开启
PRINT_MULTI_GPU_CLOVER_BISTABCG 逐迭代打印）运行 C++ 参考 Clover BiStabCG，
解析 ##LOOP 逐迭代残差（norm2），保存为 JSON。

用法:
    source ./env.sh
    LD_LIBRARY_PATH=/tmp/qcu-printbcg:$LD_LIBRARY_PATH python examples/qcu/dev73_5_bcg_conv.py \
        --lattice 8 16 16 16 --dtype c64
"""
import torch, os, sys, re, json, argparse
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, argv, set_ptrs

REPO = "/root/PyQCU"
LOG_DIR = os.path.join(REPO, "logs", "dev73_5")
os.makedirs(LOG_DIR, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lattice', nargs=4, type=int, default=[8,16,16,16])
    ap.add_argument('--dtype', default='c64', choices=['c64','c128'])
    ap.add_argument('--mass', type=float, default=0.05)
    ap.add_argument('--atol', type=float, default=1e-6)
    args = ap.parse_args()
    Lx,Ly,Lz,Lt = args.lattice
    DT = define._LAT_C128_ if args.dtype == 'c128' else define._LAT_C64_

    # C++ 用 std::cout 打印（fd 1），需在 OS 层重定向捕获
    import os, tempfile
    tmpf = tempfile.TemporaryFile(mode='w+')
    old_fd = os.dup(1)
    os.dup2(tmpf.fileno(), 1)

    params[define._LAT_X_]=Lx; params[define._LAT_Y_]=Ly
    params[define._LAT_Z_]=Lz; params[define._LAT_T_]=Lt
    params[define._LAT_XYZT_]=Lx*Ly*Lz*Lt
    params[define._GRID_X_],params[define._GRID_Y_],params[define._GRID_Z_],params[define._GRID_T_]=1,1,1,1
    params[define._PARITY_]=0; params[define._NODE_RANK_]=0; params[define._NODE_SIZE_]=1
    params[define._DAGGER_]=0; params[define._MAX_ITER_]=1000
    params[define._DATA_TYPE_]=DT
    params[define._SET_INDEX_]=0; params[define._SET_PLAN_]=1
    params[define._VERBOSE_]=0; params[define._SEED_]=42; params[define._TEST_IN_CPU_]=0
    av = argv.to(dtype=define.dtype(DT).to_real())
    av[define._MASS_]=args.mass; av[define._ATOL_]=args.atol; av[define._SIGMA_]=0.1
    dt = define.dtype(DT); ls = define.lat_shape(params)
    g=torch.zeros([2,3,3,4]+ls,dtype=dt,device='cuda')
    fi=torch.randn([2,4,3]+ls,dtype=dt,device='cuda'); fo=torch.zeros_like(fi)
    ce=torch.zeros([4,3,4,3]+ls,dtype=dt,device='cuda'); cei=torch.zeros_like(ce)
    coo=torch.zeros_like(ce); coi=torch.zeros_like(ce)
    params[define._SET_INDEX_]=0; params[define._SET_PLAN_]=-1
    qcu.applyInitQcu(set_ptrs, params, av); qcu.applyGaussGaugeQcu(g,set_ptrs,params)
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=2; params[define._PARITY_]=0
    qcu.applyInitQcu(set_ptrs, params, av); qcu.applyCloversQcu(ce,cei,g,set_ptrs,params)
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=2; params[define._PARITY_]=1
    qcu.applyInitQcu(set_ptrs, params, av); qcu.applyCloversQcu(coo,coi,g,set_ptrs,params)
    params[define._SET_INDEX_]+=1; params[define._SET_PLAN_]=1
    qcu.applyInitQcu(set_ptrs, params, av)
    qcu.applyCloverBistabCgQcu(fo,fi,g,ce,coo,cei,coi,set_ptrs,params)
    torch.cuda.synchronize()
    os.dup2(old_fd, 1); os.close(old_fd)
    tmpf.seek(0)
    raw = tmpf.read()
    tmpf.close()

    # 解析逐迭代残差（norm2 -> sqrt）
    curves = {}
    lines = raw.splitlines()
    residuals = []
    for ln in lines:
        m = re.search(r'##LOOP:(\d+)##Residual\(norm2\):([-\d.eE+]+)', ln)
        if m:
            residuals.append(float(m.group(2)))
    conv = [abs(r)**0.5 for r in residuals]   # sqrt(norm2)，逐迭代
    iters = len([r for r in residuals if r > args.atol**2])
    curves['residual_norm2'] = residuals
    curves['residual_norm'] = conv
    curves['iterations'] = len(conv)
    curves['iters_to_atol'] = iters
    curves['final_residual'] = abs(conv[-1]) if conv else None
    # 只取前 N 行原始输出作日志
    log_lines = [ln for ln in lines if '##RANK' in ln]
    log_path = os.path.join(LOG_DIR, f"bistabcg_{Lx}x{Ly}x{Lz}x{Lt}_{args.dtype}_convergence.log")
    with open(log_path, 'w') as f:
        f.write("\n".join(log_lines))
    json_path = os.path.join(LOG_DIR, f"bistabcg_{Lx}x{Ly}x{Lz}x{Lt}_{args.dtype}_conv.json")
    with open(json_path, 'w') as f:
        json.dump(curves, f, indent=2)
    print(f"[BiStabCG conv] lattice={Lx}x{Ly}x{Lz}x{Lt} {args.dtype} "
          f"iters={iters} total={len(conv)} final_res={curves['final_residual']:.3e}")
    print(f"  saved -> {json_path}")


if __name__ == "__main__":
    main()
