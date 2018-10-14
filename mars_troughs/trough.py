"""This has the trough object with all the necessary functions.
"""
import numpy as np
import cffi
import glob
import os
here = os.path.dirname(__file__)+"./"
#Both of these are temporary until interpolators are implemented in C
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import RectBivariateSpline as RBS

mars_troughs_dir = os.path.dirname(__file__)
include_dir = os.path.join(mars_troughs_dir,'include')
lib_file = os.path.join(mars_troughs_dir,'_mars_troughs.so')
# Some installation (e.g. Travis with python 3.x) name
# this e.g. _mars_troughs.cpython-34m.so,
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
        self.params = np.array(params)
        self.model_number = model_number
        #Load in supporting data
        insolation, ins_times = np.loadtxt(here+"/Insolation.txt", skiprows=1).T
        ins_times = -ins_times #positive times are now in the past
        self.insolation = insolation
        self.ins_times  = ins_times
        #Create the look up table for retreats
        self.lags       = lags = np.arange(16)+1
        self.lags[0] -= 1
        self.lags[-1] = 20
        self.retreats   = np.loadtxt(here+"/R_lookuptable.txt").T
        #Splines
        self.ins_spline = IUS(ins_times, insolation)
        self.iins_spline = self.ins_spline.antiderivative()
        self.ins2_spline = IUS(ins_times, insolation**2)
        self.iins2_spline = self.ins2_spline.antiderivative()
        self.ret_spline = RBS(self.lags, self.ins_times, self.retreats)
        self.re2_spline = RBS(self.lags, self.ins_times, self.retreats**2)
        #Initialize the C splines
        #Broken for now, since we need to C-order everything
        #_lib.initialize_basic_splines(self.ins_times, len(self.ins_times), self.insolation, self.lags, len(self.lags), self.retreats, 1)
        pass

    def set_model(self, params):
        assert len(params)==len(self.params), \
            "New and original parameters must have the same shape."
        self.params = params
        return
        
    def get_insolation(self, time):
        return self.ins_spline(time)
        
    def get_retreat(self, lag, time):
        #Change to a library call
        return self.ret_spline(time, lag)

    def get_accumulation(self, time):
        #Model dependent
        num = self.model_number
        p = self.params
        if num == 0:
            a = p[1:2] #a*Ins(t)
            return -1*(a*self.ins_spline(time))
        if num == 1:
            a,b = p[1:3] #a*Ins(t) + b*Ins(t)^2
            return -1*(
                a*self.ins_spline(time)+
                b*self.ins2_spline(time))
        pass

    def get_integ_accumulation(self, time):
        #This is the depth the trough has traveled
        #Model dependent
        num = self.model_number
        p = self.params
        if num == 0:
            a = p[1:2] #a*Ins(t)
            return -1*(
                a*(self.iins_spline(time) - self.iins_spline(0)))
        if num == 1:
            a,b = p[1:3] #a*Ins(t) + b*Ins(t)^2
            return -1*(
                a*(self.iins_spline(time) - self.iins_spline(0))+
                b*(self.iins2_spline(time) - self.iins2_spline(0)))
        pass

    def get_trajectory(self):
        #Model dependent
        pass

    def lnlikelihood(self):
        #Model dependent
        pass

if __name__ == "__main__":
    print("Test main() in trough.py")
    test_params = [-1, 1e-8]
    model_number = 0
    tr = Trough(test_params, model_number)
    
    import matplotlib.pyplot as plt
    times = tr.ins_times
    fig, ax = plt.subplots(ncols=1, nrows=3, sharex=True)
    ax[0].set_ylabel("Insolation")
    ax[1].set_ylabel("-Accumulation")
    ax[2].set_ylabel(r"$y(t) = \int_0^t{\rm d}t\ A(t)$")
    ax[-1].set_xlabel(r"Lookback Time (yrs)")
    ax[1].ticklabel_format(style='scientific', scilimits=(0,0), useMathText=True)
    ax[2].ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)
    ax[0].plot(times, tr.get_insolation(times))
    ax[1].plot(times, tr.get_accumulation(times))
    ax[2].plot(times, tr.get_integ_accumulation(times))
    
    plt.show()
