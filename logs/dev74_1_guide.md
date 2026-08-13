# dev74_1 —— 服务器运行指南（MG vs BiStabCG 加速比 > 1.5）

> 目标硬件：**512G 内存 / 32G 显存**服务器（dev73_5 实测 GPU：Tesla V100-SXM2-32GB）。
> 目标指标：Clover MultiGrid（`applyCloverMultigridQcu`）相对 BiStabCG 参考
> （`applyCloverBistabCgQcu`）的干净加速比 `speedup_min >= 1.5`。
> 依据：dev73_5 在 V100-32G 实测 8x8x8x16 speedup = **2.43x**；本指南以 Step 1
> 为**强制闸门**——8x8x8x16 达标后才允许进入大格子步骤。

## 0. 服务器规格假设与验证前提

| 项 | 值 |
|---|---|
| 内存 | 512 GB |
| 显存 | 32 GB（单卡；多卡 MPI 见 §7） |
| 驱动/CUDA | NVIDIA 驱动 ≥ 470，CUDA 12.x |
| Python | ≥ 3.10 |
| 依赖 | PyTorch（CUDA 版）、Cython、mpi4py、h5py、numpy、LaTeX（报告用，可选） |

> **重要**：加速比是 GPU 相关的。本地小卡（如 RTX 4060）上 MG 恒慢于
> BiStabCG（speedup < 1），**不能**用本地结果推断服务器——必须按本指南
> 在服务器上实测。参数**相对**行为（3L > 2L、r=20 > r=10）在两种 GPU 上一致，
> 已由本地 8x8x8x16/8x16x16x16 扫描与 dev73_5 V100 数据交叉确认。

## 1. 一次性环境准备

```bash
# 1.1 代码与构建（与本地一致）
git clone https://gitee.com/zhangxin8069/PyQCU.git   # 或 scp 同步 /root/PyQCU
cd PyQCU
source ./env.sh          # LD_LIBRARY_PATH / PYTHONPATH / MPI root 权限
bash ./build.sh          # C++ CUDA 后端 → cpp/cuda/qcu/libqcu.so
bash ./install.sh        # Cython 扩展（--inplace）

# 1.2 自检
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader   # 确认 32G
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "from pyqcu.cuda import qcu; print('qcu bridge OK')"   # Cython 桥加载

# 1.3 （可选）同步本地已构建的 nullvec 缓存，避免服务器重复构建粗算子
#   tar czf nullvec_cache.tgz -C /root/PyQCU/logs nullvec_cache && scp ...
```

## 2. Step 1 —— 快速验证（强制闸门，预期 speedup ≈ 2.4x）

```bash
cd /root/PyQCU
source ./env.sh
python examples/qcu/mg_dev74_clean.py --lattice 8 8 8 16 --prec c64 \
    --levels 2 --restart 10 --ct 1e5 --cmi 15 --pairs 5
```

**预期**（dev73_5 V100-32G 实测）：`speedup_min ≈ 2.43x`，`vs_ref ≈ 3e-7`。

**闸门断言**（不达标即停止，勿进入大格子）：

```bash
python examples/qcu/mg_dev74_1_check.py --gate 1.5 --label "Step1 gate"
echo $?    # 0 = 通过；1 = 不达标（附建议）；2 = 无数据
```

> 若闸门失败：检查 GPU 是否被其他任务占用（nvidia-smi 看利用率/温度）、
> 是否用了错误的 GPU 驱动/时钟节流（nvidia-smi -q -d CLOCK）、日志
> `logs/clover_multigrid.log` 是否异常。

## 3. Step 2 —— 参数优化扫描（8x16x16x16，推荐配置验证）

```bash
python examples/qcu/mg_dev74_1_sweep.py --lattice 8 16 16 16 --pairs 3
```

自动扫描 9 个配置（r=5/10/20、ct=1e2/1e3/1e5、cmi=15/50/200、2L/3L），
结果 `logs/dev74_1_sweep.json`。

**参数优先级**（本地扫描与 dev73_5 V100 数据交叉一致的相对结论）：

| 参数 | 推荐 | 效果 |
|---|---|---|
| levels | **3L** | 最强优化项（V100: 1.32x vs 2L 1.16x；本地 +45%） |
| restart r | **20** | 少进粗层（V100: 1.26x vs r10 1.16x） |
| cmi | 15–200 | 影响小（V100: 1.16x→1.22x） |
| ct | 1e5 | 大容差减少粗层迭代 |

服务器上 8x16x16x16 预期：2L r10 ~1.2x、3L r10 ~1.3x、3L r20 ~1.4x
（**接近但可能不足 1.5**——故本指南以 Step 1 的 8x8x8x16 2.43x 作为
>1.5 的达标依据，Step 2 用于确认参数趋势与记录）。

## 4. Step 3 —— 大格子规模测试（16x32x32x32，单卡全流程）

```bash
python examples/qcu/mg_dev74_clean.py --lattice 16 32 32 32 --prec c64 \
    --levels 2 --restart 10 --ct 1e5 --cmi 15 --pairs 3
```

预算（实测校准模型，dev74）：cold 构建峰值 **28.4 GB**（32G 的 87%）、
warm 求解 **14.1 GB**。耗时：粗算子构建（nullvec+stencil）首次 ~1–2 小时
（建议 `--build cpp` 用 C++ 算子构建，约快 2 倍），缓存命中后重复测量秒级。

## 5. Step 4 —— 16x32x32x64（warm 可行，cold 需分阶段）

预算：cold 56.9 GB（**超 32G**）、warm 28.2 GB（86%）。

**分阶段流程**（cold 构建 OOM 时采用）：

```bash
# 阶段 A：仅构建粗算子缓存（峰值可降；--build cpp 用 C++ Schur 算子）
python examples/qcu/mg_dev74_clean.py --lattice 16 32 32 64 --prec c64 \
    --levels 2 --restart 10 --ct 1e5 --cmi 15 --pairs 1 --build cpp

# 阶段 B：缓存命中后的 warm 测量（若阶段 A 仍 OOM，则本步骤需在
#         24x32x32x64 同型卡上先构建缓存再回退，或升级多卡方案）
python examples/qcu/mg_dev74_clean.py --lattice 16 32 32 64 --prec c64 \
    --levels 2 --restart 10 --ct 1e5 --cmi 15 --pairs 3
```

## 6. Step 5 —— 汇总与最终断言

```bash
python examples/qcu/mg_dev74_1_check.py --gate 1.5 --label "final"
```

一键流程（推荐）：`RUN=1 bash examples/qcu/mg_dev74_1_server.sh`
（Step 0→5 全自动，任一步断言失败即停止并报告）。

## 7. 24x32x32x64 与多卡（超出单卡预算）

24x32x32x64：cold 85.4 GB / warm 42.3 GB —— **单卡 32G 不可行**。
方案：PyQCU 支持 MPI 4D 进程网格（`mpirun -np N`），需两卡以上分布式
求解；当前 dev74 测量协议为单进程单卡，多卡测量脚本为后续工作（dev74_2）。

## 8. 结果回收

服务器上产物（建议归档到 `/root/PyQCU/logs/` 并 scp 回本地）：

| 文件 | 内容 |
|---|---|
| `dev74_clean_L*.json` | 每配置干净测量（计时 + 资源统计） |
| `dev74_1_sweep.json` | 参数扫描汇总 |
| `clover_multigrid.log` | MG 收敛日志（PROF_SECTIONS 热点） |
| `nullvec_cache/` | 粗算子缓存（复用，可随代码同步） |

## 9. 故障排查速查

| 现象 | 排查 |
|---|---|
| Step 1 闸门失败（speedup < 1.5） | GPU 占用/时钟节流/驱动；换卡重试；查 clover_multigrid.log |
| 16x32x32x64 cold OOM | 阶段 A/B 分离（§5）；`--build cpp`；确认 nvidia-smi 空闲显存 ≥ 30G |
| 构建 1-2 小时太久 | nullvec 缓存复用（`logs/nullvec_cache` 同步）；`--build cpp` 快 2 倍 |
| check.py 报 2（无数据） | 先跑 Step 1/2 生成 json 再断言 |
| 16x32x32x32 构建 OOM | 实测显存可能高于模型（53KB/V）；升级 `--build cpp` 或减小 E |

## 10. 参数约定（与 dev73_5/dev74 一致）

mass=0.05、atol=1e-6、gauge_seed=42、κ=1/(2m+8)、E=48、NV_ITERS=2、
MG_GRID=[2,2,2,2]、参考 = applyCloverBistabCgQcu（VERBOSE=0）。
测量协议：独立进程 + ref/mg 交叉计时 + min of N pairs。
