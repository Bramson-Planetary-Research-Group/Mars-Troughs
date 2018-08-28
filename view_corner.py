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

nwalkers = len(ml_params)*2+2
nburn = nwalkers*100

fig = corner.corner(chain[nburn:])
fig.savefig("chains/corner.png")
plt.show()
