---
name: md-tex-sync
description: |
  当用户要求 Markdown 与 LaTeX/PDF 保持同步、把大型 Markdown/TeX 文档整合成
  全文全集、修复公式/表格/伪代码在 Markdown 或 PDF 中的显示错误、生成源码
  快照与全附件 SHA256 清册，或说“md 和 tex 同步”“同源编译”“双格式交付”
  “全附件全集”“文档渲染整改”时使用。
metadata:
  openclaw:
    emoji: 🔄
---

# md-tex-sync — Markdown/LaTeX/PDF 同源与全集归档

把可编辑 Markdown 作为唯一内容源，生成可编译 LaTeX 和可交付 PDF；同时把原始
源文档、附件、图表和数据清册纳入可哈希、可重建、可审阅的归档结构。

## 执行前置

遵循当前目录 `AGENTS.md` 的「技能执行公共契约」。先完成只读盘点，再修改 Markdown、
生成 TeX 或编译 PDF；不得从渲染结果反推源码或覆盖原始证据。

## 核心原则

1. **Markdown 是唯一内容源**：正文、公式、算法、表格、来源和附件索引先写入
   Markdown。TeX 与 PDF 是派生件，不维护第二套人工正文。
2. **先校验正确性，再追求美观**：每个关键公式和算法先对当前源码核验变量、符号、
   更新顺序、归一化和适用条件，再处理排版。
3. **不得用 Markdown 宏包直接吞完整长文**：大型文档中的 `$...$`、`$$...$$`、
   下划线、Setext 标题和 HTML details 会被错误重解释。应由确定性转换器保护数学、
   代码和表格，再生成独立 LaTeX。
4. **原始全文与裁决正文分层**：裁决正文保证可读；原始源文件逐字快照、重复内容
   按 SHA256 去重并保留 aliases；附件与数据建立完整哈希清册。
5. **不自我引用**：Markdown 不把由其生成的 `.tex/.pdf` 纳入自身附件哈希；
   TeX 头注释记录 Markdown 源 SHA256，作为同步锚点。
6. **公式和伪代码不得降级为纯文本**：公式使用 MathJax/KaTeX 兼容语法；算法保留
   输入、初始化、循环、分支、停止条件、输出、断点和源码锚点。
7. **图必须先保证真实，再保证清晰**：优先嵌入原始 PNG/SVG/PDF，保留图题、单位、
   基线、样本数、范围和来源。超宽图使用原向量链接或拆分，不整页缩小到不可读。
8. **渲染闸门必须实测**：PDF 两遍 XeLaTeX 后要求 `Overfull=0`、`Float too large=0`、
   缺字 `=0`、TeX 错误 `=0`；全部页面可栅格化，无空白页和边缘裁切。
9. **构建必须幂等可追踪**：重复运行不得改变 Markdown 与 TeX 内容；PDF 允许编译
   时间元数据变化。每次交付记录三种文件的 SHA256、页数、文件大小和验证命令。

## 触发时机

- 用户要求“md 与 tex/PDF 同步”“同源维护”“双格式交付”“编译 PDF”。
- 用户要求修复公式、表格、算法、图片或代码块在 Markdown/PDF 中的显示错误。
- 用户要求把多份文档整合成“全文全集”“全附件”“全数据”，并保留来源与冲突裁决。
- 用户要求把大型 Markdown 转成 LaTeX，且不能依赖在线 Pandoc、MathJax 或浏览器。
- 与其他技能配合：内容裁决用 `analy/pure`，源码核验用 `debug/review`，
  排版验收用 `report`，最终归档和技能索引用 `init/skill-creator/form`。

## 工作流程

### Step 1. 锁定权威源与交付范围

1. 建立范围清单：Markdown 主源、LaTeX 派生源、PDF、图片、源码附件、数据表、
   Office/PDF 资产和构建脚本。
2. 为每份文件记录：路径、日期、SHA256、大小、类型、生成/源文件关系。
3. 同一主题有多个版本时，按日期、修订链、当前源码和明确撤回声明确定权威版本。
4. 对重复版本只保留最新权威版；旧版独特证据进入历史章节或被撤回结论表。
5. 明确“不纳入本次哈希”的生成物，例如 `.aux/.log/.out`、逐页渲染缓存和
   Markdown 自己生成的 `.tex/.pdf`。

### Step 2. 建立内容正确性基线

1. 从当前源码核验公式的变量、左右乘顺序、dagger、site、parity 和归一化。
2. 每个算法拆成可验证字段：

```text
输入 → 初始化 → 循环 → 分支/更新 → 断点检测 → 停止条件
→ 输出 → 真残差/不变量 → 源码锚点
```

3. 对关键公式执行三类检查：

| 检查 | 目的 |
|---|---|
| 量纲与极限 | 自由场、零场、大质量、退化维度 |
| 对称性 | gamma5、Hermitian、adjoint、parity |
| 实现一致性 | 源码变量位置、更新顺序、默认值、dtype |

4. 公式与源码不一致时，先改正公式与说明，再生成任何一种交付格式。

### Step 3. 完善 Markdown 主源

1. 正文、裁决、附录和来源台账全部写入同一 Markdown。
2. 行内公式使用 `$...$`，展示公式使用独占行的 `$$ ... $$`；不要使用
   `\[...\]`，因为部分 Markdown 解析器会把它当作转义方括号。
3. 伪代码使用四反引号或波浪线围栏，保留缩进、循环和停止条件。
4. 表格避免在数学单元中使用裸 `|`；向量范数使用 `\|x\|` 等无冲突写法。
5. 每一段关键结论附来源路径、版本、口径、单位、误差和限制。
6. 在文件末尾设置生成区标记：

```text
<!-- BEGIN GENERATED FULL ARCHIVE -->
<!-- END GENERATED FULL ARCHIVE -->
```

### Step 4. 生成全文与全附件附录

1. `docs/**` 中所有独立 `.md/.tex` 源按 SHA256 去重。
2. 每个唯一内容只嵌入一次，其他相同路径列入 aliases。
3. 原始行尾空格和 tab 若为了可读展示而转为可见标记，必须同时保留原始 SHA256
   和路径，并在注释中明确这是“可读视图”，不是字节级副本。
4. 所有 PDF、PNG、PPTX、Office XML、脚本、日志和数据文件进入附件清册：

```text
path | bytes | extension | SHA256
```

5. 附件按 `docs 顶层`、`data`、`assets`、`Office`、`其他` 分组，避免单张表
   溢出。
6. 排除生成缓存、`__pycache__`、`.pyc`、LaTeX aux/log/out、逐页渲染图和
   主 Markdown 自身生成的 `.tex/.pdf`，防止哈希循环和构建振荡。

### Step 5. Markdown 到 LaTeX 转换

1. 使用确定性转换器，不使用不识别数学或 HTML details 的 Markdown 宏包直接处理
   完整长文。
2. 解析优先级：

| 对象 | 处理 |
|---|---|
| `$$...$$` | 保护为占位符，再输出 `displaymath` |
| `$...$` | 保护为占位符，避免 `_` 被解析成 emphasis |
| fence/code | 逐字写入 `Verbatim` 或 listings |
| table | 转成 `longtable`，p 列、ragged-right、按列数缩放字号 |
| image | 读取并解码 URL 编码路径，设置宽度和最大高度 |
| HTML `<br>` | 转为 TeX 换行 |
| `<details>` | 不作为正文交给 TeX；原文改由附录输入 |

3. 公式内只使用 MathJax/KaTeX 广泛支持的宏；优先 `\mathrm`、`\|`、`\approx`，
   避免旧式 `\rm`、不可移植字体宏和中文 `\text{...}`。
4. 大型源码原文使用 `\lstinputlisting` 或 `\VerbatimInput` 进入附录，避免把
   原始 TeX 与生成 TeX 混在同一解析层。
5. TeX 头必须写入：

```tex
% Source: <markdown-path>
% Source SHA256: <sha256>
```

### Step 6. 编译与版式闸门

1. 使用 XeLaTeX 两遍编译，包含 shell escape 仅用于已审计的确定性生成器。
2. 编译日志逐项检查：

```bash
grep -c 'Overfull' build.log
grep -c 'Float too large' build.log
grep -c 'Missing character' build.log
grep -c '^!' build.log
```

四项必须全部为零。

3. 渲染全部页面并统计：

```bash
pdftoppm -png -r 50 report.pdf page
find . -name 'page-*.png' | wc -l
```

4. 页面图像执行空白和边缘检查；内容不得触碰安全边距，不能把不可读缩小作为
   溢出修复。
5. 对超宽表依次采用：减少装饰列 → `p` 列换行 → `seqsplit` 长 token →
   缩小一档字号 → 语义拆表；不得直接整页横缩。
6. 对图依次采用：原图重绘 → 裁 panel → 分页 → 保留原向量链接；不遮盖刻度、
   图例和单位。

### Step 7. 同步哈希与幂等检查

1. 完成构建后比较：

```bash
sha256sum report.md report.tex report.pdf
```

2. 从 TeX 头读取 `Source SHA256`，必须与当前 Markdown 一致。
3. 再次运行 `--tex-only` 或等价生成步骤，Markdown 与 TeX 的哈希必须不变。
4. PDF 内容一致性以页数、文本抽取、图片路径和日志闸门验证；PDF 时间戳允许变化。
5. 若幂等失败，检查是否存在 `__pycache__`、构建时间、随机 ID、绝对临时路径或
   Markdown 自身生成的 TeX/PDF 被纳入内部哈希。

### Step 8. 交付与技能同步

1. 交付 Markdown、TeX、PDF 和确定性构建脚本四个实体。
2. 在 Markdown 的“同源交付物”表写明四者职责和统一重建命令。
3. 将可复用经验写入项目 `skills/` 与 `/root/configure/skills/`，两份技能内容
   保持一致。
4. 更新两套 `skills/AGENTS.md` 的技能表、计数和必要的互鉴表。
5. 最终执行 Markdown 链接检查、公式分隔符检查、TeX 编译、全页渲染、SHA256
   同步和 `git diff --check`。

## PyQCU 已实现示例

当前实现位于：

```text
/root/PyQCU/docs/High-Performance_Implementation_MG_assets/build_full_documentation.py
```

它完成：

- 刷新 Markdown 全文快照与全附件 SHA256 清册；
- 将同一 Markdown 正文转为独立 LaTeX；
- 排除生成 `.tex/.pdf`、`__pycache__` 和字节码的自引用；
- 两遍 XeLaTeX 编译；
- 将 Markdown SHA256 写入 TeX 头；
- 对 565 页 PDF 执行 Overfull、缺字、错误、空白页和边缘裁切闸门。

统一重建命令：

```bash
cd /root/PyQCU/docs
python High-Performance_Implementation_MG_assets/build_full_documentation.py
```

## 错误处理

| 场景 | 根因 | 处理 |
|---|---|---|
| `$` 公式被截断 | Markdown 把 `_`、`&` 或 Setext 标题重解释 | 解析前保护数学占位符，显示公式转 `displaymath` |
| `\[...\]` 丢失反斜杠 | Markdown 把 `\[` 当转义字符 | Markdown 只用 `$$`，构建时再转换 |
| 表格列错位 | 数学单元含裸 `|` | 改 `\|`、`\lVert` 或把公式移出表格 |
| 算法缩进丢失 | 围栏被当普通段落 | 使用 fence 并转 `Verbatim/listings` |
| 图片路径含 `%` 或空格 | URL 编码和 TeX 参数冲突 | 构建前 `unquote`，路径用 `\detokenize` 或安全宏 |
| 字体缺字 | 代码/正文所用字体不含符号 | 切换等宽字体、登记 Unicode fallback，仍缺则记录 |
| Overfull | 长路径、长 hash、宽表或超大图 | seqsplit、p 列、语义拆表、换用全页浮动图 |
| 缺 `昇`、箭头、框线 | 单一 CJK/mono 字体无该字形 | 选择有对应 glyph 的字体；构建后缺失字符必须归零 |
| Markdown 哈希每次变化 | 清单包含生成物或 `__pycache__` | 排除派生文件与运行缓存，只对稳定源哈希 |
| TeX 可编译但语义错误 | 公式未按源码核验 | 回到 Step 2，建立公式/源码对照表 |
| 页面空白或裁切 | 浮动体过大或安全区估算错误 | 全页栅格化，回收浮动体，降低图高并保留图题 |
| PDF 很大 | 原文附录和完整清单已内嵌 | 保留结构，压缩图片分辨率；不删除关键正文 |

## 注意事项

- 不以“能编译”代替“内容正确”；公式、算法和性能口径必须有源码或数据证据。
- 不以“页数少”代替“内容完整”；关键公式、算法、单位、误差和限制不得省略。
- 不把生成 PDF 的时间戳、临时目录路径或构建随机数写入 Markdown。
- 不删除原始证据来换取排版整洁；重复内容先去重，再保留 aliases。
- 破坏性清理、批量重命名和跨仓库覆盖必须先获得明确授权。
- 对超长文档优先语义拆页，不用任意缩放、极小字号或裁切隐藏问题。
