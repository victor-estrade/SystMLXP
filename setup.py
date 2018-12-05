# coding: utf-8
from __future__ import division

from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "Cython facilities",
    ext_modules = cythonize('merging.pyx'),  # accepts a glob pattern
)
