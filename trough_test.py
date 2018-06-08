import numpy as np
import scipy.optimize as op
import scipy.interpolate as interp
import scipy.integrate as integ
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, "./src/")
import C_trough_model as trough_model

#Read in the data
data = np.loadtxt("RealXandZ.txt")
x_data,z_data = data
x_data *= 1000 #Change x from km to m

#Read in the insolation and mess with it a bit
ins, times = np.loadtxt("Insolation.txt", skiprows=1).T
max_ins = np.max(ins)
times[1:] = -times[1:]
ins = ins/max_ins #Normalize it
ins = -(ins-1) #flip it so it's all positive

#Define an array of lags for which the Retreat is known for
lags = np.arange(16)+1
lags[0] -= 1
lags[-1] = 20

#Read in the retreats
R = np.loadtxt("R_lookuptable.txt")
print R.shape

#Times for which we want x(t) and z(t)
ts = np.linspace(min(times), times[-10], len(times)*100)

#Define the initial positions
xi, zi = 0, 0

#Guess some basic parameters
beta=0.001 #Acc 1
gamma=0.0 #Acc 2
l = 13 #lag parameter
params = np.array([beta, gamma, l])

#Test call
x, z = trough_model.get_trough_path(xi, zi, ts, params, times, ins, lags, R)
print "Test call complete"

def cost(params):
    be, ga, l = params
    if be < 0 or ga < 0.0: return 1e99 #Disallow wrong signs
    if l < 0 or l > 20: return 1e99 #Only let lag be in this range
    #if lt < 0 or denom < 0: return 1e99 #frequencies can't be negative
    x_out, z_out = trough_model.get_trough_path(xi, zi, ts, params, times, ins, lags, R)
    if max(x_out) < max(x): return 1e99 #Force the model to go far enough in +x direction
    z_spline = interp.interp1d(x_out, z_out)
    
    #Find the cost of each data point
    #ci = (z[1:] - z_spline(x[1:]))**2 #differene squared
    #ci = np.fabs(z[1:] - z_spline(x[1:])) #absolute difference
    ci = np.fabs((z[1:] - z_spline(x[1:]))/z_spline(x[1:])) #fractional difference
    #Return the sum of the individual costs
    return np.sum(ci)

#A first guess
beta=0.001
gamma=0.01
l = 7 #lag
lt = times[-1]/2
denom = lt
guess = np.array([beta, gamma, l])
print "Test call to cost function:"
print cost(guess)

#The returned value is a `result` object
result = op.minimize(cost, x0=guess, method='Nelder-Mead')
print result

#Pull out the best result and plot it
pars = result.x
x, z = trough_model.get_trough_path(xi, zi, ts, pars, times, ins, lags, R)
plt.plot(x_data, z_data, marker='o', ls='-', c='k')
plt.plot(x, z, ls=':', c='r')
#plt.gcf().savefig("trough_example.png", dpi=400, bbox_inches='tight')
plt.show()
