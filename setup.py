from os import path
import subprocess
from distutils.core import Extension, setup
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy
class CMakeBuild(build_ext):
    def run(self):
        subprocess.check_call(['bash','./setup.sh'])
        super().run()
qcu_home = path.abspath("./qcu")
VERSION = "0.0.1"
LICENSE = "MIT"
DESCRIPTION = "Python wrapper for qcu written in Cython."
extensions = [
    Extension(
        "pyqcu.pointer",
        ["./pointer.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c",
    ),
    Extension(
            "pyqcu.pyqcu",
            ["./pyqcu.pyx"],
            include_dirs=["./include", numpy.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            library_dirs=[qcu_home],
            libraries=["qcu"],
            language="c",
        )
]
ext_modules = cythonize(
    extensions,
    language_level="3",
)
packages = [
    "pyqcu",
]
package_dir = {
    "pyqcu": ".",
}
setup(
    name="pyqcu",
    version=VERSION,
    description=DESCRIPTION,
    author="ZhangXin8069",
    author_email="zhangxin8069@qq.com",
    packages=packages,
    ext_modules=ext_modules,
    license=LICENSE,
    package_dir=package_dir,
    cmdclass={'build_ext': CMakeBuild},
)
