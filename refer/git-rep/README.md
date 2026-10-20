# PyQCU 参考 Git 库

本目录以 Git submodule 固定五个上游格点 QCD 项目。参考源码以上游仓库为准，
`refer/git-rep/<name>/` 工作树只读；PyQCU 自产分析报告统一保存在 `docs/`。

## 上游仓库

| 路径 | 上游来源 | 固定提交 | 用途 |
|---|---|---|---|
| `DDalphaAMG` | <https://github.com/mrottmann/DDalphaAMG> | `ddbdf0f2aa6586e6a8d357d6f89ff92a7ad071d0` | Wilson-Clover 代数多重网格求解器 |
| `DDalphaAMG-SM` | <https://github.com/JFNCastellanos/DDalphaAMG-SM> | `abbccf125bddbedb2f84a2c1bc06eead6003a887` | 二维 Schwinger 模型版代数多重网格求解器 |
| `PyQUDA` | <https://github.com/CLQCD/PyQUDA> | `138d8ad6e4d6e0805ecc7ec0a59a45febe0482ce` | QUDA 的 Python/Cython 封装与工具 |
| `quda` | <https://github.com/lattice/quda> | `e75ee8835f7c93683c6c96975eddc2e476c13154` | NVIDIA GPU 格点 QCD 库 |
| `kuqcd` | <https://atomgit.com/kunpengcompute/kuqcd> | `0d86e74a103180c507907b6cf6722758c6aac4ab` | 鲲鹏 CPU 格点 QCD 参考实现 |

这些上游项目为 PyQCU 的算子、求解器、多重网格和性能对照实现提供了重要参考，
在此感谢各项目及其贡献者。

## 管理约定

```bash
# 初始化或恢复全部参考库
git submodule update --init --recursive -- refer/git-rep

# 查看当前固定提交
git submodule status --recursive -- refer/git-rep
```

- 不修改 submodule 工作树中的上游文件，也不把上游文档复制进主仓库。
- 更新参考版本时，先在上游仓库确认提交，再更新 gitlink、`.gitmodules` 和本文件。
- 需要实验补丁或裁剪构建时，在 `data/quda-*` 等独立工作树中进行，并通过补丁文件记录
  相对上游固定提交的变化。
- `docs/` 仅保存 PyQCU 自产的分析、纯化剖析和技术报告；每份报告同时保留 TeX 源和 PDF。

## 分析报告

`docs/` 当前包含 34 份文件，按上游项目分组：

| 项目 | 报告集合 |
|---|---|
| DDalphaAMG | `analy_ddamg_20260817`、`pure_ddamg_20260817`、`pure_qcd_amg_20260815`、`report_ddalphaamg_multigrid_20260828` |
| DDalphaAMG-SM | `analy_ddamg_sm_20260817`、`pure_ddamg_sm_20260817` |
| PyQUDA | `analy_pyquda_20260817`、`pure_pyquda_20260817`、`pure_pyquda_core_20260815` |
| QUDA | `analy_multigrid_20260824`、`analy_multigrid_impl_20260824`、`analy_quda_20260817`、`pure_quda_20260817`、`pure_quda_core_20260815`、`report_quda_multigrid_20260827`、`report_quda_multigrid_20260828`、`report_quda_multigrid_20260828_2` |

每个报告名对应一个 `.tex` 源文件和一个 `.pdf` 构建产物。
