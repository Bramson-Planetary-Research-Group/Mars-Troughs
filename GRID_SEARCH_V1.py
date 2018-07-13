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

def cost(params):
    #be, ga, l = params #constant lag model
    alpha, beta, gamma, l0, l1, l2 = params
    #if be < 0 or ga < 0.0: return 1e99 #Disallow wrong signs
    #if l < 0 or l > 20: return 1e99 #Only let lag be in this range
    #Prior for the quadratic lag model
    #lag_test = l2*1e-13*times**2 + l1*1e-7*times + l0
    #if any(lag_test < 0) or any(lag_test > 20):
    #    return 1e99
    #if lt < 0 or denom < 0: return 1e99 #frequencies can't be negative

    
    x_out, z_out = trough_model.get_trough_path(xi, zi, ts, params, times, ins, lags, R)
    if max(x_out) < max(x_data):
        print "now this one"
        return 1e99 #Force the model to go far enough in +x direction
    #if min(z_out) > min(z_data):
    #    print "this one"
    #    return 1e99 #Force the model to go far enough in +x direction
    z_spline = interp.interp1d(x_out, z_out)
    #x_spline = interp.interp1d(z_out, x_out)
    
    #Find the cost of each data point
    #ci = (z_data[1:] - z_spline(x_data[1:]))**2 #differene squared
    #ci = np.fabs(z_data[1:] - z_spline(x_data[1:])) #absolute difference
    ci = np.fabs((z_data[1:] - z_spline(x_data[1:]))/z_spline(x_data[1:])) #fractional difference
    #ci = np.fabs((x_data[1:] - x_spline(z_data[1:]))/x_spline(z_data[1:])) #fractional difference

    #Return the sum of the individual costs
    return np.sum(ci)

#Ranges for parameters
alpha_all = [-1e-6, -1e-7, -1e-8, -1e-9, -1e-10, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
beta_all = [-1e-4, -1e-5, -1e-6, -1e-7, -1e-8, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
gamma_all = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
a_all = [5e-10, 1e-10, 5-11, 1e-11, 5e-12, 1e-12, 5e-13, 1e-13, 5e-14, 1e-14]
b_all = [5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8]
c_all = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

Nal = len(alpha_all)
Nbe = len(beta_all)
Nga = len(gamma_all)
Na = len(a_all)
Nb = len(b_all)
Nc = len(c_all)
N = Nal*Nbe*Nga*Na*Nb*Nc
output = np.zeros((N, 7))
print "output size:\n\t",np.shape(output)

def do_grid():
    ind = 0
    for alpha in alpha_all:
        for beta in beta_all:
            for gamma in gamma_all:
                for a in a_all:
                    for b in b_all:
                        for c in c_all:
                            guess = np.array([alpha, beta, gamma, a, b, c])
                            output[ind, 0:-1] = guess
                            #Calculate cost function
                            output[ind, -1] = cost(guess)
                            print "Done with ind = %d"%ind
                            ind+=1
                            ###
                            #COMMENT OUT THE IF STATMENT WHEN YOU
                            #ARE READY TO RUN FOR REAL
                            ###
                            if ind > 10: return
                            continue
                        continue
                    continue
                continue
            continue
        continue

do_grid()
print "Finished. Saving in output.txt"
header = "alpha beta gamma a b c cost"
fmt = "%.1e %.1e %.1e %.1e %.1e %.1e %e "
np.savetxt("output.txt", output, header=header, fmt=fmt)
