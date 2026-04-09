from setuptools import setup, find_packages
import subprocess
from distutils.core import Extension, setup
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy
import os
HOME = os.path.dirname(os.path.abspath(__file__))
print("HOME:", HOME)


class CMakeBuild(build_ext):
    def run(self):
        subprocess.check_call(["bash", f"{HOME}/build.sh"])
        super().run()


extensions = [
    Extension(
        "pyqcu.cuda.qcu",
        [f"{HOME}/pyqcu/cuda/qcu/qcu.pyx"],
        include_dirs=[f"{HOME}/cpp/cuda/qcu/python", numpy.get_include()],
        library_dirs=[f"{HOME}/cpp/cuda/qcu"],
        libraries=["qcu"],
        language="c",
    )
]
ext_modules = cythonize(
    extensions,
    language_level="3",
)
setup(
    name="PyQCU",
    version="0.0.1",
    description="Python wrapper for QCU written in Cython.",
    author="ZhangXin8069",
    author_email="zhangxin8069@qq.com",
    packages=find_packages(exclude=["test.*"]),
    ext_modules=ext_modules,
    license="MIT",
    cmdclass={'build_ext': CMakeBuild},
    python_requires=">=3.6",
    install_requires=["mpi4py", "cython", "h5py", "torch"],
    url="https://github.com/zhangxin8069/PyQCU",
    keywords=['c++', 'cuda', 'python',
              'quantum chromodynamics(QCD)', 'lattice QCD', 'high performance computing', 'made in China'],
)
