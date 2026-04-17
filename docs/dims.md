```bash
splited_rank.shape = [x,y,z,t]
gauge.shape = [c,c,d]+[x,y,z,t]
gauge_eo.shape = [p]+[c,c,d]+[x,y,z,t] # [d]:[x,y,z,t]
fermion.shape = [s,c]+[x,y,z,t]
fermion_eo.shape = [p]+[s,c]+[x,y,z,t]
wilson_term.shape = [s,c,s,c]+[x,y,z,t]
wilson_term_eo.shape = [p]+[s,c,s,c]+[x,y,z,t]
clover_term.shape = [s,c,s,c]+[x,y,z,t]
clover_term_eo.shape = [p]+[s,c,s,c]+[x,y,z,t]
```
