"""
Python bindings for code used to model martian troughs.
"""

import cffi
import glob
import os
import numpy as np

__author__ = "Tom McClintock <tmcclintock@email.arizona.edu>"

mars_troughs_dir = os.path.dirname(__file__)
include_dir = os.path.join(mars_troughs_dir,'include')
lib_file = os.path.join(mars_troughs_dir,'_mars_troughs.so')

_ffi = cffi.FFI()
for file_name in glob.glob(os.path.join(include_dir,'*.h')):
    _ffi.cdef(open(file_name).read())
_lib = _ffi.dlopen(lib_file)

