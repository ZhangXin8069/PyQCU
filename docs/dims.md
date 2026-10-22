# PyQCU 张量布局

轴语义固定为：

- `s`：spin；
- `c`：color；
- `d`：gauge direction；
- `p`：parity；
- `x/y/z/t`：时空维，始终是最后四轴。

```text
splited_rank.shape   = [x,y,z,t]
gauge.shape          = [c,c,d,x,y,z,t]
gauge_eo.shape       = [p,c,c,d,x,y,z,t]
fermion.shape        = [s,c,x,y,z,t]
fermion_eo.shape     = [p,s,c,x,y,z,t]
wilson_term.shape    = [s,c,s,c,x,y,z,t]
wilson_term_eo.shape = [p,s,c,s,c,x,y,z,t]
clover_term.shape    = [s,c,s,c,x,y,z,t]
clover_term_eo.shape = [p,s,c,s,c,x,y,z,t]
```

parity 轴只在 EO 布局出现；full 布局不包含 `p`。HDF5 的 `zyxt` 内部顺序必须通过
项目 I/O 转换函数与运行时 `xyzt` 布局互转，不能手工重排后混用。
