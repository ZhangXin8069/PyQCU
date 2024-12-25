from os import path
from distutils.core import Extension, setup
from Cython.Build import cythonize
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess

class CMakeBuild(build_ext):
    def run(self):
        subprocess.check_call(['bash','./setup.sh'])
        super().run()

VERSION = "0.0.1"
LICENSE = "MIT"
DESCRIPTION = "pyqcu by zhangxin"
path.join("./qcu", "libqcu.so")

extensions=[
    Extension(
        "pyqcu.qcu",
        ["./pyqcu.pyx"],
        include_dirs=["./include"],
        library_dirs=["./qcu"],
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
    "pyqcu": "./",
}
package_data = {
    "pyqcu": ["*.pyi", "*.pxd", "*.py"],
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
    package_data=package_data,
    cmdclass={'build_ext': CMakeBuild},
)
