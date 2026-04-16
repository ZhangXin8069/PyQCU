from ._wilson import give_wilson as give_wilson
from ._wilson import give_wilson_eo as give_wilson_eo
from ._wilson import give_wilson_oe as give_wilson_oe
from ._wilson import give_hopping_plus as give_hopping_plus
from ._wilson import give_wilson_plus as give_wilson_plus
from ._wilson import give_hopping_minus as give_hopping_minus
from ._wilson import give_wilson_minus as give_wilson_minus
from ._clover import make_clover as make_clover
from ._clover import add_I as add_I
from ._clover import inverse as inverse
from ._clover import give_clover as give_clover
from ._clover import give_clover_ee as give_clover_ee
from ._clover import give_clover_oo as give_clover_oo
from ._operator import hopping as hopping
from ._operator import sitting as sitting
from ._operator import operator as operator
from argparse import Namespace
Namespace.__module__ = "pyqcu.dslash"
