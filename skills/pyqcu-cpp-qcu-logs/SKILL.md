---
name: pyqcu-cpp-qcu-logs
description: cpp/cuda/qcu/logs 的归档 skill：该目录已于 2026-10-15 删除（仅两文件，已入 git），此处保留其内容清单、删除依据与后端运行产物的现行落点约定。
---
# cpp/cuda/qcu/logs（已删除）

C++ CUDA 后端（`cpp/cuda/qcu`）的本地运行产物目录。**该目录连同其两个文件已于
2026-10-15 删除**（工作区状态 `D cpp/cuda/qcu/logs/AGENTS.md`、
`D cpp/cuda/qcu/logs/schur_mg_test_stdout.txt`，待随本批提交生效），仓库内不再有该实体目录。

## 删除依据

删除前该目录只有两个文件，均已被 git 跟踪（`git ls-tree -r HEAD cpp/cuda/qcu/logs/`
可回溯）：

| 文件 | 删除前的内容 |
|---|---|
| `AGENTS.md` | 该目录的库级 AGENTS 说明 |
| `schur_mg_test_stdout.txt` | 2026-08-16 的一次失败 stdout 存档（129 B），全文为<br>`python: can't open file '.../cpp/cuda/qcu/examples/qcu/conftest.schur.multigrid.py': [Errno 2] No such file or directory` |

判据：`schur_mg_test_stdout.txt` 引用的脚本路径与其 `examples/` 目录**在仓库中早已不存在**，
该文件不构成可复现记录；`AGENTS.md` 随目录一并失去意义。二者均无引用方（删除前全库
grep 无其他引用）。用户于 2026-10-15 明确确认删除。

**不要把它当作「后端证据已丢失」**：同日的失败与修复过程记录可分别用
`git show <tag>:logs/fix-report-2026-08-24.md` 与
`git show dev73:logs/dev73/review-2026-07-28*.md` 读取，本目录两个文件不是其中的一环。

## 现行约定（后端运行产物的落点）

- 构建输出（`bash ./make.sh` / 根 `build.sh` 的编译器与链接信息）**不再往 `cpp/**` 里写**；
  `cpp/**` 只保留源码与构建脚本。
- C++ 后端测试输出走现役入口 `pyqcu/testing/qcu/`（各 `conftest.*.py` 与
  `single_qcu_*.py`），收敛日志的规范路径是仓库根 `logs/clover_multigrid.log`
  （解析函数 `pyqcu/cuda/_logs.py::parse_mg_log` / `parse_convergence_histories`）。
- 当前任务报告与运行摘要暂存在仓库根 `logs/**`。先提交报告并打 tag，再在后续清理
  提交中删除工作区副本；历史读取使用 `git show <tag>:<path>`（见
  `docs/ORGANIZATION.md` 与根 `AGENTS.md` 的「文档/日志/数据」条）。
- 若将来确有需要落在 `cpp/cuda/qcu/` 下的产物，**先确认是否要入版本库**，并在
  根 `AGENTS.md` 与该目录的 `AGENTS.md` 中登记；这里曾是 ABI/构建证据的落点，
  误删或漏跟踪都会让复现链断掉。
