from setuptools import setup, find_packages
import subprocess
from distutils.core import Extension, setup
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy
class CMakeBuild(build_ext):
    def run(self):
        subprocess.check_call(['bash','./setup.sh'])
        super().run()
extensions = [
    Extension(
        "pyqcu.pointer",
        ["./src/python/pointer.pyx"],
        include_dirs=["./include/python", numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c",
    ),
    Extension(
            "pyqcu.qcu",
            ["./src/python/pyqcu.pyx"],
            include_dirs=["./include/cpp", "./include/python", numpy.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            library_dirs=["./lib"],
            libraries=["qcu"],
            language="c",
        )
]
ext_modules = cythonize(
    extensions,
    language_level="3",
)

setup(
    name="pyqcu",
    version="0.0.1",
    description="Python wrapper for qcu written in Cython.",
    author="ZhangXin8069",
    author_email="zhangxin8069@qq.com",
    packages=find_packages(exclude=["test", "test.*"]),
    ext_modules=ext_modules,
    license="MIT",
    cmdclass={'build_ext': CMakeBuild},
    python_requires=">=3.6",
    install_requires=["numpy", "scipy", "mpi4py", "cython", "cupy"],
    test_suite="test"
    
)
