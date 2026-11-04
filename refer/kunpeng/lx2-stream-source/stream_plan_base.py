"""Explicit tested axes; every configuration executes all four STREAM kernels."""
import random
DEFAULT_ARRAY=16*1024**2

def make_plan(cpus,batch):
 cases=[]
 def add(name,domains,threads,tier,kind,*,array=DEFAULT_ARRAY,targets=None,split=None,pattern=None):
  workers=[]
  for d in domains:
   for index,cpu in enumerate(cpus[d][:threads]):
    target=targets[d] if targets else d
    node=target+(16 if tier=='HBM' or (tier=='split' and index>=split) else 0)
    a=b=out=node
    if pattern:
     a=d+(16 if pattern[0]=='H' else 0);b=d+(16 if pattern[1]=='H' else 0);out=d+(16 if pattern[2]=='H' else 0)
    workers.append([cpu,a,b,out,d])
  cases.append({'name':name,'batch':batch,'kind':kind,'cpu_domains':domains,'threads_per_domain':threads,'tier':tier,'split_ddr_threads_per_domain':split,'array_bytes_per_worker':array,'workers':workers,'input_output_pattern':pattern,'memory_targets':targets})
 if batch=='numa':
  for d in range(16):
   ts=[1,36] if d not in [2,6,10,14] else [1,2,4,8,12,18,24,36,37,38]
   for nt in ts:
    for tier in ['DDR','HBM']:add(f'numa_d{d}_t{nt}_{tier}',[d],nt,tier,'numa')
 elif batch=='hierarchy':
  sets={f'die{k}':list(range(4*k,4*k+4)) for k in range(4)}
  sets.update(cpu0=list(range(8)),cpu1=list(range(8,16)),node=list(range(16)),spread4=[2,6,10,14])
  for label,domains in sets.items():
   for nt in [1,4,9,18,36]:
    for tier in ['DDR','HBM']:add(f'{label}_t{nt}_{tier}',domains,nt,tier,'hierarchy')
  for label,domains in [('same_die2',[2,3]),('same_cpu2',[2,6]),('cross_cpu2',[2,10])]:
   for tier in ['DDR','HBM']:add(f'{label}_t36_{tier}',domains,36,tier,'hierarchy')
 elif batch=='access':
  sets={**{f'numa{d}':[d] for d in [2,6,10,14]},**{f'die{k}':list(range(4*k,4*k+4)) for k in range(4)},'cpu0':list(range(8)),'cpu1':list(range(8,16)),'node':list(range(16))}
  for label,domains in sets.items():
   for nd in ([9,18,27] if label.startswith('numa') else [18]):add(f'split_{label}_{nd}_{36-nd}',domains,36,'split','split',split=nd)
  for d in [2,6,10,14]:
   for relation,t in [('same_die',d^1),('other_die',d^4),('other_cpu',d^8)]:
    for tier in ['DDR','HBM']:add(f'remote_d{d}_{relation}_{tier}',[d],36,tier,'remote',targets={d:t})
   for pattern in ['DDH','HHD','DHH','HDD']:
    add(f'cross_tier_d{d}_{pattern}',[d],36,'mixed','cross_tier',pattern=pattern)
 elif batch=='cache':
  for d in [2,10]:
   for size in [4096,16384,131072,262144,1048576,16777216]:
    for tier in ['DDR','HBM']:add(f'cache_d{d}_{size//1024}KiB_{tier}',[d],1,tier,'working_set',array=size)
 elif batch=='pilot':
  add('pilot_L1',[2],1,'DDR','pilot',array=4096)
  add('pilot_NUMA37',[2],37,'HBM','pilot')
  add('pilot_node',list(range(16)),36,'HBM','pilot')
  add('pilot_crossCPU',[2],36,'HBM','pilot',targets={2:10})
  add('pilot_crossTier',[2],36,'mixed','pilot',pattern='DHH')
  add('pilot_split',[2],36,'split','pilot',split=18)
 else:raise ValueError(batch)
 assert len({c['name'] for c in cases})==len(cases)
 for c in cases:
  assert len({w[0] for w in c['workers']})==len(c['workers']) and len(c['workers'])<=576
 random.Random(20261028).shuffle(cases)
 return cases
