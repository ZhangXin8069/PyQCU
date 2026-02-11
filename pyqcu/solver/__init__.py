from ._bistabcg import bistabcg as bistabcg
from ._multigrid import multigrid as multigrid
from ._multigrid import hopping as hopping
from ._multigrid import sitting as sitting
from ._multigrid import operator as operator
from argparse import Namespace
Namespace.__module__ = "pyqcu.solver"
