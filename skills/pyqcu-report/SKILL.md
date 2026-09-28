---
name: pyqcu-report
description: |
  当用户要求制作或修改 PyQCU 中文横版 Beamer、竖版完整讲稿或同源 PPTX，
  或说“做汇报 PPT”“生成讲稿”“横版改 LaTeX”“补充公式/算法页”时使用。
  纯性能基准、算法分析或代码修复任务不使用本技能。
metadata:
  openclaw:
    emoji: 📽️
---

# pyqcu-report — PyQCU 科研汇报生成

把仓库中的技术报告、源码、tag 数据和历史报告整理成一套内容一致、证据可追溯的
横版汇报 PDF、PPTX 和竖版讲稿。

## 核心原则

1. **横版与竖版职责分离**
   - 横版 PDF 负责公式、算法、流程图和数据图，文字只做解释。
   - 竖版讲稿负责完整叙事、公式含义、时间控制、风险和问答，不重复横版所有细节。
2. **统一视觉但不过度装饰**
   横版结构元素统一使用黑/灰/白，移除标题线、圆角框、色块、彩色 bullet 和装饰边框；
   饱和颜色只允许出现在真实数据图中。标题、页脚、算法和公式依靠字体层级与留白分组。
3. **证据优先**
   - 老版本文档只作为视觉或历史参考，事实与默认值优先读取当前源码和当前 tag 数据。
   - 性能数字优先读取当前 tag 对应的 CSV、JSON、combined 记录和生成脚本。
   - `git tag test27` 的精确口径与旧 tag 的实验结果不可混合。
   - 508 报告数据只用于“先前成果/大规模历史测试”页：测试环境为 4 张
     Pre-Wukong DCU、64--1024 MPI 进程，记录 Wilson `3.12x`、Clover `7.19x`；
     它不证明当前 Strict-MG 已完成千卡扩展，也不与 test27 速度比混算。
4. **公式和算法是主体**
   复杂算法使用编号公式和单列表格式伪代码；不要把伪代码压成普通段落。
   每个正式公式使用章节化编号（如 `(3.2)`），图和表分别使用 `图 章节.序号`、
   `表 章节.序号`，算法使用 `Algorithm 章节.序号`。
   主体页面应优先填满公式、伪代码、数据流或数据图，不把版面让给背景性段落。
5. **图必须来自真实证据**
   优先使用报告中的原始图表；自制图只能从同一 CSV/JSON 重绘，并注明来源。
   数据图必须给出基线、方向定义、样本数和限制，例如 `R=QUDA/PyQCU`。
6. **字体可投影**
   正文和公式不能靠缩小字体塞入页面。空间不足时删重复说明、合并公式行或增加页数。
   横版推荐正文不低于约 10--11 pt，图内标签保持可辨；竖版讲稿使用一倍半行距。
7. **版式验收是硬闸门**
   XeLaTeX 两遍编译后必须 `Overfull=0`、`Float too large=0`，并渲染全部页面检查
   遮挡、裁切、空白和页脚侵入。
8. **PPTX 与 PDF 同源**
   横版 PDF 作为视觉权威；若交付 PPTX，应从同一版 Beamer PDF 逐页渲染后封装，
   避免 PDF 和 PPTX 出现不同内容或样式。
9. **Strict 与 Legacy 分栏**
   - `test27` 性能链对应 Strict full-coarse MATPC + fused right-FGMRES。
   - 33 点 coarse stencil 属于 Legacy compact-Schur，不能与 Strict 的
     \(X/Y/\widehat Y\) 写成同一个粗算子。
   - Legacy CUDA 的 `local_orthogonalize` 使用 batched reduced QR；
     Strict/reference 的 aggregate CGS 是另一套实现。
   - 不应把 CUDA Graph、cooperative coarsest solve 或完全异步 MPI
     外推到所有 Strict 配置。

## 推荐字体

在 XeLaTeX 中统一配置：

```tex
\setmainfont{TeX Gyre Termes}
\setsansfont{TeX Gyre Heros}
\setmonofont{TeX Gyre Cursor}
\setCJKmainfont{FandolSong-Regular.otf}[
  BoldFont=FandolSong-Bold.otf,
  ItalicFont=FandolKai-Regular.otf
]
\setCJKsansfont{FandolHei-Regular.otf}[BoldFont=FandolHei-Bold.otf]
\setCJKmonofont{FandolSong-Regular.otf}
```

用途：正文 `FandolSong`，标题/页眉 `FandolHei`，强调 `FandolKai`，
英文和数字 `TeX Gyre Termes`，代码 `TeX Gyre Cursor`。

## 工作流程

### Step 1：锁定来源和口径

1. 先读当前源码，再读最近的相关报告；不从记忆重建机制或数值：
   - 当前性能：`docs/report_multigrid_quda_pyqcu_YYYYMMDD.tex`
   - 算法构造：`docs/report_pyqcu_mg_operator_construction_*.tex`
   - CUDA/MPI 实现：`docs/report_pyqcu_multigrid_cuda_*.tex`
   - 历史大规模成果：`docs/张鑫 508应用测试报告-PyQCU/`
   - 会话上下文：`logs/v2026*.txt`
2. 明确数据基线 tag。比较前核对 config/input、phase、真残差、显存和硬件 provenance。
3. 建立 `claim -> evidence -> visual` 清单；每个图表至少记录源文件、生成命令和限制。
4. 分离三类事实：
   - 当前正式性能；
   - 实现机制或历史实验；
   - 尚未验证的扩展方向。

### Step 2：制定 16:9 逐页结构

默认保持项目既有框架，按需增加公式或算法页：

1. 标题与范围；
2. Wilson/Clover 作用量；
3. 原始 Dslash；
4. 奇偶 Schur Dslash；
5. MultiGrid 构造；
6. 预条件 Bi-CGStab；
7. Strict `D=X+H` 与 full coarse 资产；
8. 粗层 `X/Y/Yhat` 公式与矩阵顺序；
9. V-cycle + flexible FGMRES；
10. CUDA--C++ 优化；
11. QUDA 对照协议与原图；
12--15. 性能分布、分组中位数、阶段/显存、正确性与下一步；
16. 参考实现和复现接口；
17. 508 报告的大规模先前成果；
18. 致谢。

如用户要求“内容尽量全”，宁可增加算法或公式页，也不能把两页内容强塞成一页。

### Step 3：提取报告数据图

优先保留报告原图。对 PDF 图表可在工作区生成高分辨率 PNG：

```bash
pdftoppm -png -singlefile -scale-to-x 1800 -scale-to-y -1 \
  data/report_multigrid_comprehensive_20260928/final_protocol/report/mg_peak_memory.pdf \
  docs/High-Performance_Implementation_MG_assets/report_chart_assets/mg_peak_memory
```

典型图：

- `overlap_matrix.pdf`：对照协议与分组/逐单元趋势；
- `mg_speedup_units.pdf`：66 个精确单元分布；
- `mg_stage_breakdown.pdf`：逐层阶段成本；
- `mg_peak_memory.pdf`：设备级显存；
- `mg_finest_residual.pdf`：最细层真残差；
- 508 报告 `media/image25.png`：TEST9 64--1024 进程扩展原始表。

当原图超宽或超高时，裁切某个 panel 或从对应 CSV 重绘；不要整页缩放导致标签不可读。

### Step 4：生成横版 Beamer

横版文件命名固定为：

```text
docs/<Title>_presentation.tex
docs/<Title>_presentation.pdf
```

要求：

1. 使用 `ctexbeamer` 和 `aspectratio=169`。
2. 页面使用统一的 `frame title`、页脚和色彩角色；不要每页换主题。
3. 公式密集页使用 `block` 或普通公式区；算法使用单列表格，保留缩进、分支和停止条件。
4. 图不需要 `figure` 浮动；直接在 `frame` 内 `\includegraphics` 并控制宽高。
5. 每张页只保留一个主结论。横版不写长篇论文段落。
6. 若某页需要超过约 10% 自动缩小，拆分页面，不继续压缩字体。

### Step 5：生成竖版完整讲稿

竖版文件命名固定为：

```text
docs/<Title>_speaker_script.tex
docs/<Title>_speaker_script.pdf
```

每页讲稿至少包含：

- 对应横版页码和建议时长；
- 主口播稿；
- 公式、符号或数据图的解释；
- 必要时的证据边界；
- 视觉提示或转场句。

讲稿应比横版更完整，可以放入：

- 公式推导的中间步骤；
- 数据图读数与限制；
- 常见追问的完整回答；
- 约 10 分钟主稿与可按时间删减的扩展段。

### Step 6：编译和验收

从 `docs/` 目录运行，避免相对图路径错误：

```bash
xelatex -interaction=nonstopmode -halt-on-error -file-line-error \
  -output-directory=High-Performance_Implementation_MG_assets/beamer_build \
  'High-Performance Implementation of Multigrid Solver for Lattice QCD_presentation.tex'
```

再运行第二遍并检查：

```bash
TITLE='High-Performance Implementation of Multigrid Solver for Lattice QCD'
BUILD=docs/High-Performance_Implementation_MG_assets/beamer_build
grep -c 'Overfull' "$BUILD/$TITLE.log"
grep -c 'Float too large' "$BUILD/$TITLE.log"
pdfinfo "docs/${TITLE}_presentation.pdf"
pdftoppm -png -r 90 "docs/${TITLE}_presentation.pdf" \
  docs/High-Performance_Implementation_MG_assets/beamer_render/page
```

必须满足：

- 页数与逐页账本一致；
- `Overfull=0`、`Float too large=0`；
- 全页目测无裁切、遮挡、页脚冲突；
- 公式/图/算法编号连续且可在正文引用；
- PDF 文字可提取，至少能检索标题、关键数字和算法名。

### Step 7：同步 PPTX

如果用户同时需要 PPTX，以同一 Beamer PDF 为唯一源：

```bash
python skills/pyqcu-report/scripts/beamer_pdf_to_pptx.py \
  --pdf "docs/${TITLE}_presentation.pdf" \
  --out-pptx "docs/${TITLE}.pptx" \
  --work-dir docs/High-Performance_Implementation_MG_assets/beamer_slides \
  --width 1920 --height 1080
```

脚本以 1920x1080 渲染每页并封装同源 PPTX；随后校验 OOXML 压缩包、
slide 数、image 数和 PNG hash。不要维护另一套独立 PPT 版式。

## 错误处理

| 场景 | 处理 |
|---|---|
| 图表来自旧 tag，与当前报告冲突 | 以用户指定 tag/报告为准，旧图明确标注“历史” |
| PDF 上出现 `Overfull` | 先拆公式、删重复句子、扩大图区；最后才允许轻微 shrink |
| 字体缺字 | 使用 Fandol/TeX Gyre 组合；禁止依赖系统中不存在的思源字体 |
| 图片太宽/太高 | 裁切 panel 或从 CSV 重绘；不要整页缩放 |
| 横版文字过多 | 删除背景，把细节移入竖版讲稿 |
| 竖版内容不够 | 增加公式解释、证据边界、数据解释和问答 |
| 横版/PPTX 风格不一致 | 用 Beamer PDF 重新生成 PPTX |
| 性能结论被外推 | 明确区分 MG、BiCGStab、trace-on/off、当前与历史数据 |
| 508 数据与 test27 混用 | 拆页并标注 4 卡 DCU、64--1024 进程、早期实现和硬件/版本 |

## 注意事项

- 新增页数不会破坏框架，但必须保持主线顺序和页脚总页数同步。
- 性能图优先使用 `report_multigrid_quda_pyqcu_20260928.tex` 的原始数据；
  508 报告只在先前成果页使用。
- 公式编号、图表编号和算法编号属于交付质量的一部分，不靠手工装饰。
- 每次修改横版后，竖版讲稿、PPTX 和 `.gitignore` 交付范围都要一起核对。
- 横版新增页时，同步更新竖版讲稿页码、页脚总页数、PPTX slide 数；
  如果公式来自 `report_multigrid_quda_pyqcu_20260913.tex` 等专题报告，
  必须在页脚保留对应源码/公式段落。
