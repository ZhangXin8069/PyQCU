from setuptools import setup, find_packages
import subprocess
from setuptools.command.build_ext import build_ext
import os

from pyqcu._build_backend import GPU_BUILD_BACKENDS, normalize_build_backend

HOME = os.path.dirname(os.path.abspath(__file__))
print("HOME:", HOME)

try:
    BUILD_BACKEND = normalize_build_backend(
        os.environ.get("PYQCU_BUILD_BACKEND"))
except ValueError as error:
    raise RuntimeError(str(error)) from error
print(f"PYQCU build backend: {BUILD_BACKEND}")


class CMakeBuild(build_ext):
    def run(self):
        if BUILD_BACKEND == "cuda" and \
                os.environ.get("PYQCU_SKIP_NATIVE_BUILD") != "1":
            # The CUDA bridge owns its native build lifecycle.
            subprocess.check_call(["bash", f"{HOME}/build-nv.sh"])
        elif BUILD_BACKEND == "dcu" and \
                os.environ.get("PYQCU_SKIP_NATIVE_BUILD") != "1":
            subprocess.check_call(
                ["bash", f"{HOME}/cpp/cuda/qcu/make-dcu.sh"],
                cwd=f"{HOME}/cpp/cuda/qcu")
        super().run()


ext_modules = []
if BUILD_BACKEND in GPU_BUILD_BACKENDS:
    # Keep Cython and GPU-specific imports out of the pure KUPL build path.
    from distutils.core import Extension
    from Cython.Build import cythonize
    import numpy

    extensions = [
        Extension(
            "pyqcu.cuda.qcu",
            [f"{HOME}/pyqcu/cuda/qcu/qcu.pyx"],
            include_dirs=[
                f"{HOME}/pyqcu/cuda/qcu",
                f"{HOME}/cpp/cuda/qcu/python",
                numpy.get_include(),
            ],
            library_dirs=[f"{HOME}/cpp/cuda/qcu"],
            libraries=["qcu"],
            language="c++",
        )
    ]
    ext_modules = cythonize(
        extensions,
        language_level="3",
    )


install_requires = [
    "mpi4py", "h5py", "torch", "psutil", "scipy", "matplotlib",
    "opt_einsum",
]
if BUILD_BACKEND in GPU_BUILD_BACKENDS:
    install_requires.append("cython")


setup(
    name="PyQCU",
    version="0.0.1",
    description="Python wrapper for QCU written in Cython.",
    author="ZhangXin8069",
    author_email="zhangxin8069@qq.com",
    packages=find_packages(exclude=["pyqcu.testing.*", "pyqcu.testing"]),
    package_data={"pyqcu.cuda.qcu": ["*.pyi"]},
    ext_modules=ext_modules,
    license="MIT",
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.10",
    install_requires=install_requires,
    url="https://github.com/zhangxin8069/PyQCU",
    keywords=[
        "c++", "cuda", "python", "quantum chromodynamics(QCD)",
        "lattice QCD", "high performance computing", "made in China",
    ],
)
