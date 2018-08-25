"""This file contains a trough_model object, which should act as the permanent interface with the C library. The reason for this is because we want to carry around parameter names along with the ability to call the model itself.

This also will remove the need to do things like read in the insolation and all that stuff whenever we want to use the trough code.
"""

import numpy as np
import os, sys
sys.path.insert(0, "./src/")
import C_trough_model as ctm

#Read in the insolation and mess with it a bit
ins, times = np.loadtxt("Insolation.txt", skiprows=1).T
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
R = np.loadtxt("R_lookuptable.txt")

#Times for which we want x(t) and z(t)
ts = np.linspace(min(times), times[-10], len(times)*100)

#Define the initial positions
xi, zi = 0, 0

def get_trough_path(parameters):
    """Get the trough

    """
