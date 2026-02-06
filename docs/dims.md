```bash
splited_rank.shape = [dim of x, dim of y, dim of z, dim of t] 
gauge.shape = [dim of color, dim of color, dim of space-time(xyzt), dim of x, dim of y, dim of z, dim of t]
gauge_parity.shape = [parity for even-odd ] + gauge.shape
fermion.shape = [dim of spin, dim of color, dim of x, dim of y, dim of z, dim of t]
fermion_parity.shape = [parity for even-odd ] + gauge.shape
wilson_term.shape = [dim of spin, dim of color, dim of spin, dim of color, dim of x, dim of y, dim of z, dim of t]
wilson_term_parity.shape = [parity for even-odd ] + gauge.shape
clover_term.shape = [dim of spin, dim of color, dim of spin, dim of color, dim of x, dim of y, dim of z, dim of t]
clover_term_parity.shape = [parity for even-odd ] + gauge.shape
```
