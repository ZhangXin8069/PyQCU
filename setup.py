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
        subprocess.check_call(["bash", f"{HOME}/setup.sh"])
        super().run()


extensions = [
    Extension(
        "pyqcu.qcu",
        [f"{HOME}/pyqcu/qcu/qcu.pyx"],
        include_dirs=[f"{HOME}/extern/qcu/python", numpy.get_include()],
        library_dirs=[f"{HOME}/lib"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
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
    packages=find_packages(exclude=["test", "test.*"]),
    ext_modules=ext_modules,
    license="MIT",
    cmdclass={'build_ext': CMakeBuild},
    python_requires=">=3.6",
    install_requires=["numpy", "scipy", "mpi4py", "opt_einsum", "proplot",
                      "cython", "cupy", "h5py", "matplotlib", "seaborn"],
    url="https://github.com/zhangxin8069/PyQCU",
    keywords=['c++', 'cuda', 'python',
              'quantum chromodynamics(QCD)', 'lattice QCD', 'high performance computing', 'made in China'],
)
