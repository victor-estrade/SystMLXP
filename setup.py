# coding: utf-8
from __future__ import division

import numpy as np

from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "Merging intertwinged",
    ext_modules = cythonize('merging.pyx'),  # accepts a glob pattern
    include_dirs = [np.get_include()],
    )
