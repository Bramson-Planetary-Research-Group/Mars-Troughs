import numpy as np
import scipy.optimize as op
import scipy.interpolate as interp
import scipy.integrate as integ
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, "./src/")
import C_trough_model as trough_model

#Read in the data
data = np.loadtxt("txt_files/RealXandZ.txt")
x_data,z_data = data
x_data *= 1000 #Change x from km to m

#Read in the insolation and mess with it a bit
ins, times = np.loadtxt("txt_files/Insolation.txt", skiprows=1).T
max_ins = np.max(ins)
times[1:] = -times[1:]
ins = ins/max_ins #Normalize it
ins = -(ins-1) #flip it so it's all positive

#Define an array of lags for which the Retreat is known for
lags = np.arange(16)+1
lags[0] -= 1
lags[-1] = 20

#Read in the retreats
R = np.loadtxt("txt_files/R_lookuptable.txt")
print R.shape

#Times for which we want x(t) and z(t)
ts = np.linspace(min(times), times[-10], len(times)*100)

#Define the initial positions
xi, zi = 0, 0

def cost(params):
    #be, ga, l = params #constant lag model
    be, ga, l0, l1, l2 = params
    if be < 0 or ga < 0.0: return 1e99 #Disallow wrong signs
    #if l < 0 or l > 20: return 1e99 #Only let lag be in this range
    #Prior for the quadratic lag model
    lag_test = l2*1e-13*times**2 + l1*1e-7*times + l0
    if any(lag_test < 0) or any(lag_test > 20):
        return 1e99
    #if lt < 0 or denom < 0: return 1e99 #frequencies can't be negative
    x_out, z_out = trough_model.get_trough_path(xi, zi, ts, params, times, ins, lags, R)
    #if max(x_out) < max(x_data):
    #    return 1e99 #Force the model to go far enough in +x direction
    if min(z_out) > min(z_data):
        return 1e99 #Force the model to go far enough in +x direction
    #z_spline = interp.interp1d(x_out, z_out)
    x_spline = interp.interp1d(z_out, x_out)
    
    #Find the cost of each data point
    #ci = (z_data[1:] - z_spline(x_data[1:]))**2 #differene squared
    #ci = np.fabs(z_data[1:] - z_spline(x_data[1:])) #absolute difference
    #ci = np.fabs((z_data[1:] - z_spline(x_data[1:]))/z_spline(x_data[1:])) #fractional difference
    ci = np.fabs((x_data[1:] - x_spline(z_data[1:]))/x_spline(z_data[1:])) #fractional difference

    #Return the sum of the individual costs
    return np.sum(ci)

#A first guess
beta=2.
gamma=1.
l2 = 1.#times[-1]**-2
l1 = 1.#times[-1]**-1
l0 = 1.
guess = np.array([beta, gamma, l0, l1, l2])
#Test call
x, z = trough_model.get_trough_path(xi, zi, ts, guess, times, ins, lags, R)
print "Test call complete"

print "Test call to cost function:"
print cost(guess)

print "Starting the minimize"
result = op.minimize(cost, x0=guess, method='Nelder-Mead')
print result

#Pull out the best result and plot it
pars = result.x
x, z = trough_model.get_trough_path(xi, zi, ts, pars, times, ins, lags, R)
plt.plot(x_data, z_data, marker='o', ls='-', c='k')
plt.plot(x, z, ls=':', c='r')
plt.xlim(min(x_data), max(x_data))
plt.ylim(min(z_data), max(z_data))
#plt.gcf().savefig("trough_example.png", dpi=400, bbox_inches='tight')
plt.show()
