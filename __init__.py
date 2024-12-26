try:
    from . import pyqcu as qcu
except ImportError as e:
    print(e)
from .mpi import init
