metric,ranks,grid,bound,seconds,seconds_source,iterations,relative_residual,residual_verified,job_id,note
wall_total,16,2 2 2 2,no,186.084576,script MG_PROFILE total,0,,,2236385,fine_apply=93.463424 fine_calls=48
wall_total,16,2 2 2 2,no,196.002810,script MG_PROFILE total,0,,,2236451,unmodified control; fine_apply=101.058137
wall_total,16,2 2 2 2,yes,2.062792,script MG_PROFILE total,0,,,2236485,fine_apply=1.520722 rc=0
wall_total,32,2 2 2 4,no,102.465100,row seconds,18,7.942248513387823e-06,True,,unbound; case 342.07 s
wall_total,32,2 2 2 4,yes,3.04056982207112,row seconds,18,7.942248513387823e-06,True,2236599,data/l48_prof32_bound
us_per_site_fine_apply,16,2 2 2 2,no,2.93,derived: 93.463424 s / 48 calls / 663552 sites,,,,,
us_per_site_fine_apply,32,2 2 2 4,no,3.13,derived: 49.92 s / 48 / 331776,,,,,
ratio_total,16,2 2 2 2,unbound:bound,90.2,derived: 186.084576 / 2.062792,,,,,95x quoted against the 196.002810 control
ratio_total,32,2 2 2 4,unbound:bound,33.7,derived: 102.4651 / 3.04056982207112,,,,,
ratio_fine_apply,16,2 2 2 2,unbound:bound,61.5,derived: 93.463424 / 1.520722,,,,,
per_site_verdict,16_vs_32,2 2 2 2 vs 2 2 2 4,flat,,2.93 vs 3.13 us/site,,,,,"NOT a rank-count effect: per-site cost is flat, so no scaling discontinuity exists"
root_cause,all,all,all,,,,,,,launcher omitted KUPL_NUMA_BIND=1 KUPL_MEMORY_NODE=local+16 KUPL_HUGEPAGE=1
fix,all,all,all,,,,,,,pyqcu/test/native.py::_require_kunpeng_binding refuses a multi-rank KUPL run missing any of the three
caveat_timing,all,all,all,,,,,,,seconds are only comparable within one allocation: same unchanged script measured 1.8185/3.8661/2.7584 s across three allocations
caveat_scaling,all,all,all,,,,,,,do not read this table as a parallel scaling curve; it is one configuration defect fixed
