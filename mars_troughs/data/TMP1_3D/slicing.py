#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:51:01 2023

@author: laferrierek
"""

import numpy as np
import matplotlib.pyplot as plt


loc = '/Users/laferrierek/Box Sync/Desktop/Research/Mars_MCMC_Fork/Mars-Troughs/mars_troughs/data/TMP1_3D/'
filename = 'TMP1_v2_raw_test1.csv'
X, Y, Z = np.loadtxt(loc+filename, delimiter=',', skiprows=2, unpack=True)
# X, Y, z


fig = plt.figure(figsize=(5, 5), dpi=300)
ax = fig.add_subplot(projection='3d')
ax.scatter(X/1000, Y/1000, Z)
ax.set_zlim(-1800, -2400)
ax.invert_zaxis()
ax.set_zlabel('Depth (m)')
ax.set_xlim(40, 70)
ax.set_xlabel('Distance (km)')
ax.set_ylim(-200, -235)
ax.invert_yaxis()
ax.set_ylabel('Distance (km)')
ax.view_init(20, 250)

plt.show()


plt.plot(X, Z)


#%%
from scipy.interpolate import interp1d

cubic_interpolation_model = interp1d(Y[0:27:2]/1000, Z[0:27:2], kind = "cubic")
 
# Plotting the Graph
Y_=np.linspace((Y[0:27:2]/1000).min(), (Y[0:27:2]/1000).max(), 500)
Z_=cubic_interpolation_model(Y_)

plt.plot(Y_, Z_)
plt.scatter(Y[0:27:2]/1000, Z[0:27:2])
plt.title("Plot Smooth Curve Using the scipy.interpolate.interp1d Class")
plt.xlabel("X")
plt.ylabel("Z")
plt.show()

#%%
lens= 27

fig = plt.figure(figsize=(5, 5), dpi=300)
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:lens]/1000, Y[:lens]/1000, Z[:lens])
ax.set_zlim(-1800, -2400)
ax.invert_zaxis()
ax.set_zlabel('Depth (m)')
ax.set_xlim(40, 70)
ax.set_xlabel('Distance (km)')
ax.set_ylim(-200, -235)
ax.invert_yaxis()
ax.set_ylabel('Distance (km)')
ax.view_init(20, 250)

plt.show()


plt.plot(Y[:lens], Z[:lens])

#%%
lena= 27
lens= 92
fig = plt.figure(figsize=(5, 5), dpi=300)
ax = fig.add_subplot(projection='3d')
ax.scatter(X[lena:lens]/1000, Y[lena:lens]/1000, Z[lena:lens])
ax.set_zlim(-1800, -2400)
ax.invert_zaxis()
ax.set_zlabel('Depth (m)')
ax.set_xlim(40, 70)
ax.set_xlabel('Distance (km)')
ax.set_ylim(-200, -235)
ax.invert_yaxis()
ax.set_ylabel('Distance (km)')
ax.view_init(20, 250)

plt.show()


plt.plot(X[:lens], Z[:lens])

#%%
lena= 92
lens= -1
fig = plt.figure(figsize=(5, 5), dpi=300)
ax = fig.add_subplot(projection='3d')
ax.scatter(X[lena:lens]/1000, Y[lena:lens]/1000, Z[lena:lens])
ax.set_zlim(-1800, -2400)
ax.invert_zaxis()
ax.set_zlabel('Depth (m)')
ax.set_xlim(40, 70)
ax.set_xlabel('Distance (km)')
ax.set_ylim(-200, -235)
ax.invert_yaxis()
ax.set_ylabel('Distance (km)')
ax.view_init(20, 250)

plt.show()


plt.plot(X[:lens], Z[:lens])


#%%
dirc = '/Users/laferrierek/Desktop/Mars_Troughs/Seisware_files/COSHARPS/Horizons/PB3D/forlatlong/latlongs/'
filename = 'depth2_latlong.txt'
loc = dirc+filename

X, Y, Z, TMP_latitude, TMP_longitude, x, y = np.loadtxt(loc, delimiter=',', skiprows=1, unpack=True)

lens = 1

x = np.reshape(X/1000, (13, 1226))
y = np.reshape(Y/1000, (13, 1226))
z = np.reshape(Z, (13, 1226))

fig = plt.figure(figsize=(5, 5), dpi=300)
ax = fig.add_subplot(projection='3d')

ax.scatter(X[::lens]/1000, Y[::lens]/1000, Z[::lens], c=Z[::lens])
ax.set_zlim(0, 400)
ax.invert_zaxis()
ax.set_zlabel('Depth (m)')
ax.set_xlim(-10, 300)
ax.set_xlabel('Distance (km)')
ax.set_ylim(-200, -110)
ax.invert_yaxis()
ax.set_ylabel('Distance (km)')
ax.view_init(40, 200)

plt.show()

plt.plot(X[::lens], Z[::lens])

#%% how can i do this??

dirc = '/Users/laferrierek/Desktop/Mars_Troughs/Seisware_files/COSHARPS/Horizons/PB3D/forlatlong/latlongs/'
filename = 'depth40_latlong.txt'
loc = dirc+filename

X, Y, Z, TMP_latitude, TMP_longitude, x, y = np.loadtxt(loc, delimiter=',', skiprows=1, unpack=True)

lens = 1

fig = plt.figure(figsize=(5, 5), dpi=300)
ax = fig.add_subplot(projection='3d')

ax.scatter(X[::lens]/1000, Y[::lens]/1000, Z[::lens], c=Z[::lens])
ax.set_zlim(0, 1000)
ax.invert_zaxis()
ax.set_zlabel('Depth (m)')
ax.set_xlim(-500, 500)
ax.set_xlabel('Distance (km)')
ax.set_ylim(-500, 500)
ax.invert_yaxis()
ax.set_ylabel('Distance (km)')
ax.view_init(90, 0)

plt.show()


#%%
dirc = "/Users/laferrierek/Box Sync/Desktop/Research/Mars_MCMC_fork/Mars-Troughs/mars_troughs/data/TMP1_3D/"
filename = "TMP1_v2_depth_test2_latlongs.txt"

loc = dirc+filename
X, Y, Z, lat, long = np.loadtxt(loc, delimiter=',', skiprows=1, unpack=True)


plt.plot(X, Y)
plt.ylabel("Y")
plt.xlabel("X")
plt.show()

plt.plot(lat, long)
plt.ylabel("long")
plt.xlabel("lat")
plt.show()

plt.plot(X, Z)
plt.ylabel("Z")
plt.xlabel("X")
plt.show()

plt.plot(lat, Z)
plt.ylabel("Z")
plt.xlabel("lat")
plt.show()

plt.plot(Y, Z)
plt.ylabel("Z")
plt.xlabel("Y")
plt.show()

plt.plot(long, Z)
plt.ylabel("Z")
plt.xlabel("long")
plt.show()

fig = plt.figure(figsize=(5, 5), dpi=300)
ax = fig.add_subplot(projection='3d')
ax.invert_zaxis()

ax.scatter(X/1000, Y/1000, Z, c=Z)

plt.show()

fig = plt.figure(figsize=(5, 5), dpi=300)
ax = fig.add_subplot(projection='3d')

ax.scatter(lat, long, Z, c=Z)
ax.invert_zaxis()

plt.show()

#%%
plt.plot(Y, Z, '.')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()

plt.ylabel("Z")
plt.xlabel("Y")
plt.show()

plt.plot(X, Z, '.')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()

plt.ylabel("Z")
plt.xlabel("X")
plt.show()

dist = np.sqrt(X**2 + Y**2) 
plt.plot(dist, Z)

plt.gca().invert_yaxis()

plt.ylabel("Z")
plt.xlabel("dist")
plt.show()

dist_corr = dist - np.nanmin(dist)

plt.plot(dist_corr, Z)

plt.gca().invert_yaxis()

plt.ylabel("Z")
plt.xlabel("dist corr")
plt.show()

#%% plot with err - 475 and 10 m
plt.errorbar(dist_corr[::2]/1000, Z[::2], yerr=10, xerr=(np.sqrt((475)**2+(475)**2)/2)/1000)
plt.ylabel("Z")
plt.xlabel("dist corr (km)")
#plt.ylim((0, 100))
#plt.xlim((0, 1))
plt.gca().invert_yaxis()
plt.show()

#%% correcting distance (X, Y issues?)
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 3376.2 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

lon1 = np.min(long)
lon2 = np.max(long)
lat1 = np.min(lat)
lat2 = np.max(lat)

dist_latlong = haversine(lon1, lat1, lon2, lat2)
print(dist_latlong)

dist_ll = np.zeros((len(long)))
for i in range(len(long)):
    dist_ll[i] = haversine(lon2, lat2, long[i], lat[i])

#plt.plot(dist_ll, Z)

plt.errorbar(dist_ll[::2], Z[::2], yerr=10, xerr=(np.sqrt((475)**2+(475)**2)/2)/1000)
plt.ylabel("Z")
plt.xlabel("dist corr (km)")
#plt.ylim((0, 100))
#plt.xlim((0, 1))
plt.gca().invert_yaxis()
plt.show()
#%% 
# shoudl we be using the depth corrected ones or no? i feel like this is changing the shapes

np.savetxt(dirc+"TMP1_v2_depth_test2_XY_MCMC.txt", np.fliplr(np.vstack((dist_corr/1000, -Z))).T, header="Distance (km), Depth (m)" )

np.savetxt(dirc+"TMP1_v2_depth_test2_XY_MCMC_desampled.txt", np.fliplr(np.vstack((dist_ll[::2], -Z[::2]))).T, header="Distance (km), Depth (m)" )


