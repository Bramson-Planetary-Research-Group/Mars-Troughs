"""This script is used to create a corner plot. An example of which is corner.png, which was created for the quadratic lag and quadratic insolation model.

In addition to numpy and matplotlib, you need the corner package. pip install corner should do the trick.
"""
import numpy as np
import matplotlib.pyplot as plt
import corner

chain = np.load("chains/mcmc_chain_trough.npy")
print chain.shape

likes = np.load("chains/mcmc_likes_trough.npy")
ind = np.argmax(likes)

ml_params = chain[ind]
print "Most likely parameters:\n\t", ml_params
import trough_model
trough = trough_model.trough_object()
in_params = ml_params[1:]
trough.set_parameters(in_params)
x_out, z_out = trough.get_trough_path()
#Read in the data
data = np.loadtxt("txt_files/RealXandZ.txt")
x_data,z_data = data
x_data *= 1000 #Change x from km to m
plt.plot(x_out, z_out, label="model")
err = np.sqrt(np.exp(ml_params[0])) #sqrt of variance parameter
plt.errorbar(x_data, z_data, xerr=err, label='data')
plt.legend()
plt.show()

#exit()

nwalkers = len(ml_params)*2+2
nburn = nwalkers*100

fig = corner.corner(chain[nburn:])
fig.savefig("chains/corner.png")
plt.show()
