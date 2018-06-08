import numpy as np
import ctypes, os, sys, inspect
from ctypes import c_double, c_int, POINTER, cdll

#This is to allow this to work from anywhere, in case we implement a setup.py
library_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/c_troughs.so"
cflib = cdll.LoadLibrary(library_path)

def get_trough_path(xi, zi, ts, parameters, times, insolation, lags, retreats):
    """Get the trough path from the driver.

    Args:
        xi: initial x position in m
        zi: initial z position in m
        ts: times that we want the modeled path
        parameters: array containing the free parameters of the model
        times: array containing the times that we have the insolation

    Returns:
        x: array containing all x positions
        z: array containing all z positions

    """
    #Outline the argument types
    driver = cflib.driver
    driver.argtypes = [c_double, c_double, #xi, zi
                       POINTER(c_double), c_int, #parameters, N_parameters
                       POINTER(c_double), c_int, #times_out, N_times_out
                       POINTER(c_double), c_int, #times, N_times
                       POINTER(c_double), POINTER(c_double), #x, z
                       POINTER(c_double), #insolations
                       POINTER(c_double), c_int, #lags, N_lags
                       POINTER(c_double)] #retreats
                       
    #Create an array for x and z
    N = len(ts)
    Nt = len(times)
    x = np.zeros(N)
    z = np.zeros(N)
    Np = len(parameters)
    Nlags = len(lags)

    #Create pointers for the input arrays
    ts = np.ascontiguousarray(ts)
    ts_in = ts.ctypes.data_as(POINTER(c_double))
    times = np.ascontiguousarray(times)
    times_in = times.ctypes.data_as(POINTER(c_double))
    x_in = x.ctypes.data_as(POINTER(c_double))
    z_in = z.ctypes.data_as(POINTER(c_double))
    parameters = np.ascontiguousarray(parameters)
    params_in = parameters.ctypes.data_as(POINTER(c_double))
    insolation = np.ascontiguousarray(insolation)
    ins_in = insolation.ctypes.data_as(POINTER(c_double))
    lags = np.ascontiguousarray(lags, dtype=np.float64)
    lags_in = lags.ctypes.data_as(POINTER(c_double))
    retreats = np.ascontiguousarray(retreats.flatten())
    retreats_in = retreats.ctypes.data_as(POINTER(c_double))
    
    #Call the driver
    driver(xi, zi, params_in, Np, ts_in, N, times_in, Nt, x_in, z_in, ins_in, lags_in, Nlags, retreats_in)
    
    #Return x and z
    return x, z
