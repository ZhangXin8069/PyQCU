"""Match selected prior cases exactly to separate node variation from kernel changes."""
from stream_plan_base import make_plan as original
import random

def make_plan(cpus,batch):
 assert batch=='anchors'
 wanted=set()
 for d in [2,6,10,14]:
  for tier in ['DDR','HBM']:
   for t in [1,8,18,36]:wanted.add(f'numa_d{d}_t{t}_{tier}')
 for scope,t in [('die0',36),('die1',36),('die2',36),('die3',36),('cpu0',36),('cpu1',36),('node',36),('spread4',36),('die0',9),('spread4',9)]:
  for tier in ['DDR','HBM']:wanted.add(f'{scope}_t{t}_{tier}')
 for d in [2,6,10,14]:
  for n in [9,18,27]:wanted.add(f'split_numa{d}_{n}_{36-n}')
  for tier in ['DDR','HBM']:wanted.add(f'remote_d{d}_other_cpu_{tier}')
  for pattern in ['DDH','HHD','DHH','HDD']:wanted.add(f'cross_tier_d{d}_{pattern}')
 wanted.add('split_node_18_18')
 for d in [2,10]:
  for tier in ['DDR','HBM']:
   for k in [4,128,256]:wanted.add(f'cache_d{d}_{k}KiB_{tier}')
 cases=[c for batch in ['numa','hierarchy','access','cache'] for c in original(cpus,batch) if c['name'] in wanted]
 assert len(cases)==len(wanted)==101
 random.Random(20261028).shuffle(cases)
 return cases
