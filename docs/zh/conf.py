"""Sphinx configuration for the Chinese PyQCU site.

The documentation build deliberately does not import pyqcu: the package may
require a native CUDA/KUPL library, while the site only documents its public
interfaces.
"""

project = "PyQCU（中文文档）"
copyright = "2026, ZhangXin8069"
author = "ZhangXin8069"

version = "0.1"
release = "0.1.8"

extensions = ["myst_parser"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
language = "zh_CN"
html_search_language = "zh"

html_theme = "sphinx_rtd_theme"
exclude_patterns = ["_build"]
