"""This script is used to create a corner plot. An example of which is corner.png, which was created for the quadratic lag and quadratic insolation model.

In addition to numpy and matplotlib, you need the corner package. pip install corner should do the trick.
"""
import numpy as np
import matplotlib.pyplot as plt
import corner

chain = np.load("mcmc_chain_trough.npy")
print chain.shape

fig = corner.corner(chain)
fig.savefig("corner.png")
plt.show()
