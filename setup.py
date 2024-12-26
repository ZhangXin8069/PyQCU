from os import path, environ
import subprocess
from distutils.core import Extension, setup
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy
class CMakeBuild(build_ext):
    def run(self):
        subprocess.check_call(['bash','./setup.sh'])
        super().run()

VERSION = "0.0.1"
LICENSE = "MIT"
DESCRIPTION = "Python wrapper for qcu written in Cython."
ld_library_path = [path.abspath(_path) for _path in environ["LD_LIBRARY_PATH"].strip().split(":")]
ld_library_path.append(path.abspath("./qcu"))

print(ld_library_path)
BUILD_QCU = False
for libqcu_path in ld_library_path:
    if path.exists(path.join(libqcu_path, "libqcu.so")):
        BUILD_QCU = True
        break
else:
    import warnings
    warnings.warn("Cannot find libqcu.so in LD_LIBRARY_PATH environment.", RuntimeWarning)
extensions = [
    Extension(
        "pyqcu.pointer",
        ["./pointer.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c",
    ),
]
if BUILD_QCU:
    extensions.append(
        Extension(
            "pyqcu.pyqcu",
            ["./pyqcu.pyx"],
            include_dirs=["./", numpy.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            library_dirs=[ld_library_path],
            libraries=["qcu"],
            language="c",
        )
    )
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
package_data = {
    "pyqcu": ["*.pyi", "*.pxd", "src.pxd"],
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
