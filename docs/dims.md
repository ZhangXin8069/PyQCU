```bash
splited_rank.shape = [t,z,y,x]
gauge.shape = [c,c,d]+[t,z,y,x]
gauge_eo.shape = [c,c,d]+[p]+[t,z,y,x] # [d]:0123:x_y_z_t
fermion.shape = [s,c]+[t,z,y,x]
fermion_eo.shape = [p]+[s,c]+[t,z,y,x]
wilson_term.shape = [s,c,s,c]+[t,z,y,x]
wilson_term_eo.shape = [p]+[s,c,s,c]+[t,z,y,x]
clover_term.shape = [s,c,s,c]+[t,z,y,x]
clover_term_eo.shape = [p]+[s,c,s,c]+[t,z,y,x]
```
