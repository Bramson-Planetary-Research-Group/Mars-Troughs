import numpy as np
import scipy.optimize as op
import scipy.interpolate as interp
import scipy.integrate as integ
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, "./src/")
import C_trough_model as trough_model
import emcee #used for MCMC analyses

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

#Times for which we want x(t) and z(t)
ts = np.linspace(min(times), times[-10], len(times)*100)

#Define the initial positions
xi, zi = 0, 0

#Define the log-posterior probability of our model given the data
#Mathematically this looks like: lnPost = lnPrior + lnLike
def lnprior(params):
    """Log of the prior on the parameters.
    None right now.
    """
    return 0

def lnlike(params):
    """Log of the likelihood of the data given the parameters.
    This is what the cost function is based off of.

    NOTE: the first parameter is always the model variance, or the
    square of the errorbars on the data points.
    """
    #Currently using the XX model
    model_var, alpha, beta, gamma, a, b, c = params

    #Calculate the trough path
    in_params = params[1:]
    x_out, z_out = trough_model.get_trough_path(xi, zi, ts, in_params, times, ins, lags, R)
    if max(x_out) < max(x_data):
        return -1e99 #Force the model to go far enough in +x direction
    z_spline = interp.interp1d(x_out, z_out)
    #x_spline = interp.interp1d(z_out, x_out)

    #Compute the log likelihood, which is just -0.5*chi2 and return the sum
    LL = -0.5*(z_data[1:] - z_spline(x_data[1:]))**2/model_var
    return LL.sum()

def lnpost(params):
    #compute the prior
    lpr = lnprior(params)
    if not np.isfinite(lpr): return -1e99
    return lpr + lnlike(params)

#A first guess
model_var = 400. #meters^2
alpha = -1e-10
beta  = -1e-8
gamma = 1e-4
a = 5e-12
b = 5e-6
c = 1
guess = np.array([model_var, alpha, beta, gamma, a, b, c])

#Set up the walkers in the MCMC
nwalkers = len(guess)*2+2
nsteps = 1000
ndim = len(guess)
pos = [guess + 1e-3*guess*np.random.randn(ndim) for k in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, threads=4)
print "Running MCMC"
sampler.run_mcmc(pos, nsteps)
print "\tMCMC complete, saving chains"
np.save("chains/mcmc_chain_trough", sampler.flatchain)
np.save("chains/mcmc_likes_trough", sampler.flatlnprobability)
