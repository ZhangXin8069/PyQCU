"""Compatibility facade for the canonical :mod:`pyqcu.torch.dslash`."""

from pyqcu.torch.dslash import give_clover as give_clover
from pyqcu.torch.dslash import give_clover_ee as give_clover_ee
from pyqcu.torch.dslash import give_clover_oo as give_clover_oo
from pyqcu.torch.dslash import add_I as add_I
from pyqcu.torch.dslash import cut_I as cut_I
from pyqcu.torch.dslash import give_hopping_minus as give_hopping_minus
from pyqcu.torch.dslash import give_hopping_plus as give_hopping_plus
from pyqcu.torch.dslash import give_wilson as give_wilson
from pyqcu.torch.dslash import give_wilson_eo as give_wilson_eo
from pyqcu.torch.dslash import give_wilson_minus as give_wilson_minus
from pyqcu.torch.dslash import give_wilson_oe as give_wilson_oe
from pyqcu.torch.dslash import give_wilson_plus as give_wilson_plus
from pyqcu.torch.dslash import hopping as hopping
from pyqcu.torch.dslash import inverse as inverse
from pyqcu.torch.dslash import make_clover as make_clover
from pyqcu.torch.dslash import operator as operator
from pyqcu.torch.dslash import sitting as sitting

from argparse import Namespace

Namespace.__module__ = "pyqcu.dslash"

__all__ = [
    "give_clover",
    "add_I",
    "cut_I",
    "give_clover_ee",
    "give_clover_oo",
    "give_hopping_minus",
    "give_hopping_plus",
    "give_wilson",
    "give_wilson_eo",
    "give_wilson_minus",
    "give_wilson_oe",
    "give_wilson_plus",
    "hopping",
    "inverse",
    "make_clover",
    "operator",
    "sitting",
]
