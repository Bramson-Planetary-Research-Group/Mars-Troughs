"""This file contains a trough_model object, which should act as the permanent interface with the C library. The reason for this is because we want to carry around parameter names along with the ability to call the model itself.

This also will remove the need to do things like read in the insolation and all that stuff whenever we want to use the trough code.
"""

import numpy as np
import os, sys
sys.path.insert(0, "./src/")
import C_trough_model as ctm

class trough_object(object):

    def __init__(self):
        #Read in the insolation and mess with it a bit
        ins, times = np.loadtxt("txt_files/Insolation.txt", skiprows=1).T
        max_ins = np.max(ins)
        times[1:] = -times[1:]
        ins = ins/max_ins #Normalize it
        ins = -(ins-1) #flip it so it's all positive
        #This step above doesn't actually need to be done if we parameterize the
        #accumulation correctly.
        #Define an array of lags for which the Retreat is known for
        lags = np.arange(16)+1
        lags[0] -= 1
        lags[-1] = 20
        #Read in the retreats
        R = np.loadtxt("txt_files/R_lookuptable.txt")
        #Times for which we want x(t) and z(t)
        ts = np.linspace(min(times), times[-10], len(times)*100)
        #Set important things to local variables
        self.times = times
        self.ts = ts
        self.ins = ins
        self.lags = lags
        self.R = R

    def set_parameters(self, parameters):
        """Call this function to set parameters 
        used to calculate a migration path.
        Note: the parameters cannot include a variance term.
        """
        self.params = parameters
        
    def get_trough_path(self):
        """Get the trough path.
        """
        in_params = self.params
        xi,zi = 0, 0
        ts = self.ts
        times = self.times
        ins = self.ins
        lags = self.lags
        R = self.R
        #Initial positions
        xi, zi = 0, 0
        x_out, z_out = ctm.get_trough_path(xi, zi, ts, in_params, times, ins, lags, R)
        return [x_out, z_out]

if __name__ == "__main__":

    #Try it out
    alpha = -2e-10
    beta  = -1e-8
    gamma = 4e-4
    a = 3e-12
    b = 2e-6
    c = 1
    guess = np.array([alpha, beta, gamma, a, b, c])

    trough = trough_object()
    trough.set_parameters(guess)
    x_out, z_out = trough.get_trough_path()

    import matplotlib.pyplot as plt
    plt.plot(x_out, z_out, label="model")


    #Read in the data
    data = np.loadtxt("txt_files/RealXandZ.txt")
    x_data,z_data = data
    x_data *= 1000 #Change x from km to m
    plt.plot(x_data, z_data, label='data')
    plt.legend()
    plt.show()
