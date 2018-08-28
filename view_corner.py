"""This script is used to create a corner plot. An example of which is corner.png, which was created for the quadratic lag and quadratic insolation model.

In addition to numpy and matplotlib, you need the corner package. pip install corner should do the trick.
"""
import numpy as np
import matplotlib.pyplot as plt
import corner
import matplotlib.cm as cm
plt.rc("errorbar", capsize=3)

#Read in the data
data = np.loadtxt("txt_files/RealXandZ.txt")
x_data,z_data = data
x_data *= 1000 #Change x from km to m

def color_by_age(x, z):
    #x and y must have same length as times
    _, times = np.loadtxt("txt_files/Insolation.txt", skiprows=1).T
    times[1:] = -times[1:]
    ts = np.linspace(min(times), times[-10], len(times)*100)
    print len(times)
    #inds = (z > np.min(z_data))*(x < np.max(x_data))
    inds = (z > -500)*(x < 3e4)
    x = x[inds]
    z = z[inds]
    t = ts[inds]
    step = 20
    plot_colorline(x[::step], z[::step], t[::step])
    return

def plot_colorline(x,y,c):
    c = cm.jet((c-np.min(c))/(np.max(c)-np.min(c)))
    ax = plt.gca()
    for i in np.arange(len(x)-1):
        #if i == 0: label="model"
        #else: label = None
        ax.plot([x[i],x[i+1]], [y[i],y[i+1]], c=c[i])#, label=label)
        print i
    return

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
#plt.plot(x_out, z_out, label="model")
color_by_age(x_out, z_out)
#plt.plot(x_out, z_out, )

err = np.sqrt(np.exp(ml_params[0])) #sqrt of variance parameter
plt.errorbar(x_data, z_data, xerr=err, label='data', c='k', marker='o')
#plt.legend()
plt.xlim(0, 3e4)
plt.ylim(-500, 0)
plt.show()

exit()

nwalkers = len(ml_params)*2+2
nburn = nwalkers*100

fig = corner.corner(chain[nburn:])
fig.savefig("chains/corner.png")
plt.show()
