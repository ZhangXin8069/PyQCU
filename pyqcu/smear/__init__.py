"""Compatibility facade for the canonical :mod:`pyqcu.torch.smear`."""

from pyqcu.torch.smear import stout_smear as stout_smear
from pyqcu.torch.smear import wuppertal_smear as wuppertal_smear

from argparse import Namespace

Namespace.__module__ = "pyqcu.smear"

__all__ = ["stout_smear", "wuppertal_smear"]
