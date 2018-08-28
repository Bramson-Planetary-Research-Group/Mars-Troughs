"""This file contains functions to facilitate plotting trough trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
plt.rc("errorbar", capsize=3)
plt.rc("font", size=12, family='serif')
plt.rc("text", usetex=True)

#Read in the data
data = np.loadtxt("txt_files/RealXandZ.txt")
x_data,z_data = data
x_data *= 1000 #Change x from km to m

def color_by_age(x, z):
    #x and y must have same length as times
    _, times = np.loadtxt("txt_files/Insolation.txt", skiprows=1).T
    times[1:] = -times[1:]
    ts = np.linspace(min(times), times[-10], len(times)*100)
    print len(times), min(times), max(times)
    #inds = (z > np.min(z_data))*(x < np.max(x_data))
    inds = (z > -500)*(x < 3e4)
    x = x[inds]
    z = z[inds]
    t = ts[inds]
    step = 20
    plot_colorline(x[::step], z[::step], t[::step])
    return

def plot_colorline(x,y,c):
    Norm = colors.Normalize(vmin=np.min(c), vmax=np.max(c))
    SM = cm.ScalarMappable(norm=Norm, cmap="jet")
    SM.set_array(c)
    #cs = [cmap(ci) for ci in c]
    cs = cm.jet((c-np.min(c))/(np.max(c)-np.min(c)))
    ax = plt.gca()
    for i in np.arange(len(x)-1):
        #if i == 0: label="model"
        #else: label = None
        ax.plot([x[i],x[i+1]], [y[i],y[i+1]], c=cs[i], zorder=-1)#, label=label)
        print i
    fig = plt.gcf()
    #cbax = fig.add_axes([0.80, 0.15, 0.02, 0.7]) # cax=cbax,
    cb = fig.colorbar(SM, orientation='vertical', label=r"Trough Age [years]")
    #plt.colorbar(cax=cbax, orientation='vertical', label="Trough Age")
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

plt.xlabel("Horizonal migration [m]")
plt.ylabel("Vertical migration [m]")
#plt.legend()
plt.xlim(0, 3e4)
plt.ylim(-500, 0)
plt.show()
