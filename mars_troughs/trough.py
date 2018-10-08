"""This has the trough object with all the necessary functions.
"""
import numpy as np
import cffi
import glob
import os
#Both of these are temporary until interpolators are implemented in C
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import interp2d

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
        #Load in supporting data
        insolation, ins_times = np.loadtxt("Insolation.txt", skiprows=1).T
        ins_times = -ins_times #positive times are now in the past
        self.insolation = insolation
        self.ins_times  = ins_times
        #Create the look up table for retreats
        self.lags       = lags = np.arange(16)+1
        self.lags[0] -= 1
        self.lags[-1] = 20
        self.retreats   = np.loadtxt("R_lookuptable.txt").T
        #Temporary splines
        self.ins_spline = IUS(ins_times, insolation)
        self.ret_spline = interp2d(self.ins_times, self.lags, self.retreats)
        pass

    def set_model(self, params):
        assert len(params)==len(self.params), \
            "New and original parameters must have the same shape."
        self.params = params
        return
        
    def get_insolation(self, time):
        return self.ins_spline(time)

    def get_retreat(self, lag, time):
        return self.ret_spline(lag, time)

    def get_accumulation(self):
        #Model dependent
        pass

    def get_trajectory(self):
        #Model dependent
        pass

    def lnlikelihood(self):
        #Model dependent
        pass

if __name__ == "__main__":
    print("test")
    tr = Trough([0], 0)
    tmin, tmax = tr.ins_times[0], tr.ins_times[-1]
    times = np.linspace(tmin, tmax, 10000)

    try:
        import matplotlib.pyplot as plt
        plt.plot(times, tr.get_insolation(times))
        plt.show()
    except Exception:
        print("Install matplotlib to visualize the test.")
