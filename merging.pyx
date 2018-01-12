# coding: utf-8
from __future__ import division

import numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
INT = np.int
FLOAT = np.float
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t INT_t
ctypedef np.float_t FLOAT_t

cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
def merge_abscisse_fill_idx(np.ndarray[FLOAT_t, ndim=1] absc, 
                            np.ndarray[INT_t, ndim=1] idx, 
                            np.ndarray[FLOAT_t, ndim=1] merged_absc):
    assert absc.dtype == FLOAT and idx.dtype == INT and merged_absc.dtype == FLOAT
    cdef INT_t c = 0
    cdef INT_t n = np.shape(absc)[0]
    cdef INT_t size = merged_absc.shape[0]
    for i in range(size):
        if c < n - 1 and absc[c] < merged_absc[i] :
            c = c + 1 
        idx[i] = c
