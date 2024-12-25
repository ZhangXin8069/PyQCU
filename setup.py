from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import subprocess
import sys

class CMakeBuild(build_ext):
    def run(self):
        # 使用 CMake 构建 CUDA 扩展
        subprocess.check_call(['bash','./setup.sh'])
        super().run()

# 定义扩展模块
ext_modules = [
    Extension(
        name='pyqcu',
        sources=[]  # 不需要额外的 Python 源文件，CMake 会处理
    )
]

setup(
    name='qcu',
    version='0.0.0',
    ext_modules=ext_modules,
    cmdclass={'build_ext': CMakeBuild},
)
