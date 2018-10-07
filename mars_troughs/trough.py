"""This has the trough object with all the necessary functions.
"""
import cffi
import glob
import os

mars_troughs_dir = os.path.dirname(__file__)
include_dir = os.path.join(mars_troughs_dir,'include')
lib_file = os.path.join(mars_troughs_dir,'_mars_troughs.so')
# Some installation (e.g. Travis with python 3.x) name this e.g. _mars_troughs.cpython-34m.so,
# so if the normal name doesn't exist, look for something else.
if not os.path.exists(lib_file):
    alt_files = glob.glob(os.path.join(os.path.dirname(__file__),'_mars_troughs*.so'))
    if len(alt_files) == 0:
        raise IOError("No file '_mars_troughs.so' found in %s"%mars_troughs_dir)
    if len(alt_files) > 1:
        raise IOError("Multiple files '_mars_troughs*.so' found in %s: %s"%(mars_troughs_dir,alt_files))
    lib_file = alt_files[0]

_ffi = cffi.FFI()
for file_name in glob.glob(os.path.join(include_dir,'*.h')):
    _ffi.cdef(open(file_name).read())
_lib = _ffi.dlopen(lib_file)

class Trough(object):
    def __init__(self, params, model_number):
        """Constructor for the trough object.

        Args:
            params (array like): model parameters
            model_number (int): index of the model
        """
        self.params = params
        self.model_number = model_number
        #Here, possibly load in auxiliary data
        pass

    def set_model(self, params):
        assert len(params)==len(self.params), \
            "New and original parameters must have the same shape."
        self.params = params
        return
        
    def get_insolation(self, t):
        pass

    def get_lag(self):
        pass

    def get_retreat(self):
        pass

    def get_accumulation(self):
        pass

    def get_trajectory(self):
        pass

    def lnlikelihood(self):
        pass
