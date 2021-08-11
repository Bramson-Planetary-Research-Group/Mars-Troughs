#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 20:37:17 2021

@author: kris
"""
import numpy as np
from mars_troughs import DATAPATHS
from mars_troughs.datapaths import (load_obliquity_data,
                                    load_retreat_data)
import corner
import pickle
import matplotlib.pyplot as plt
import argparse
import os

p=argparse.ArgumentParser(description="filename for plotting")
p.add_argument("-filename",type=str,help="filename for loading mcmc object")
p.add_argument("-nmodels",type=int,help="nmodels for ensamble")
args=p.parse_args()


#Load data
xdata,ydata=np.loadtxt(DATAPATHS.TMP, unpack=True) #Observed TMP data
xdata=xdata*1000 #km to m 
(obliquity,obl_times) = load_obliquity_data() #Insolation data and times
obl_times=-obl_times
obl_times[0]=1e-10
times, retreats, lags = load_retreat_data()
times=-times
times[0]=1e-10

infile=open(args.filename+'obj','rb')
newmcmc=pickle.load(infile)
infile.close()

#create folder for saving figures
if not os.path.exists(newmcmc.directory+'figures/'):
    os.makedirs(newmcmc.directory+'figures/')

#set parameters for plotting
paramsList=list(newmcmc.tr.all_parameter_names)
numparams=len(paramsList)

lagparamsList= [string for string in paramsList if 'lag_' in string]
numlagparams=len(lagparamsList)
accparamsList= [string for string in paramsList if 'acc_' in string]
numaccparams=len(accparamsList)

nmodels=args.nmodels
totalsteps=newmcmc.sampler.iteration
initmodel=totalsteps-nmodels

#all parameters per iter
all_samples=newmcmc.sampler.get_chain()
ensemble=all_samples[initmodel:,:,:]

plt.figure()

xaxis=np.arange(initmodel+1,totalsteps+1)
for i in np.arange(1,numparams):
    plt.subplot(numparams,1,i)
    plt.plot(xaxis,ensemble[:,:,i-1])
    plt.xticks([], [])
    plt.title(paramsList[i-1])
    
plt.subplot(numparams,1,numparams)
plt.plot(xaxis,ensemble[:,:,numparams-1])
plt.title(paramsList[numparams-1])
plt.xlabel('Step')

#create folder for saving figure
if not os.path.exists(newmcmc.directory+'figures/'+'paramsIter/'):
    os.makedirs(newmcmc.directory+'figures/'+'paramsIter/')
    
plt.savefig(newmcmc.directory+'figures/'+'paramsIter/'
            +newmcmc.modelName+'_'+str(newmcmc.maxSteps)+'.pdf',
            facecolor='w',pad_inches=0.1)

#corner plot
#reshape ensemble
ensemble2d=ensemble[:,0,:]

#labels
for i in range(1,newmcmc.nwalkers):
    ensemble2d=np.vstack([ensemble2d,ensemble[:,i,:]])
    
#plot
fig=corner.corner(ensemble2d,labels=paramsList)

#create folder for saving figure
if not os.path.exists(newmcmc.directory+'figures/'+'corner/'):
    os.makedirs(newmcmc.directory+'figures/'+'corner/')
    
plt.savefig(newmcmc.directory+'figures/'+'corner/'
            +newmcmc.modelName+'_'+str(newmcmc.maxSteps)+'.pdf',
            facecolor='w',pad_inches=0.1)

#log prob
all_logprob=newmcmc.sampler.get_log_prob()
logprob=all_logprob[initmodel:,:]

plt.figure()
for i in np.arange(0,newmcmc.nwalkers):
    plt.plot(xaxis,logprob[:,i])
plt.xlabel('Step')
plt.ylabel('log prob')


#create folder for saving figure
if not os.path.exists(newmcmc.directory+'figures/'+'logprob/'):
    os.makedirs(newmcmc.directory+'figures/'+'logprob/')
    
plt.savefig(newmcmc.directory+'figures/'+'logprob/'
            +newmcmc.modelName+'_'+str(newmcmc.maxSteps)+'.pdf',
            facecolor='w',pad_inches=0.1)

plt.figure()
#autocorrelation values
autoxaxis=newmcmc.subIter*np.arange(1,totalsteps/newmcmc.subIter+1)
autoxaxis=autoxaxis[:len(newmcmc.autocorr)]

plt.plot(autoxaxis,autoxaxis/50,"--k",label=r'$\tau$<steps/50 ')
plt.plot(autoxaxis,newmcmc.autocorr[np.nonzero(newmcmc.autocorr)],label=r'$\tau$ estimate')
plt.xlabel('Step')
plt.ylabel(r'mean $\tau$')
ax=plt.gca()
ax.legend()


#create folder for saving figure
if not os.path.exists(newmcmc.directory+'figures/'+'autocorr/'):
    os.makedirs(newmcmc.directory+'figures/'+'autocorr/')
    
plt.savefig(newmcmc.directory+'figures/'+'autocorr/'
            +newmcmc.modelName+'_'+str(newmcmc.maxSteps)+'.pdf',
            facecolor='w',pad_inches=0.1)

#lag, acc rate and y per time for each model 
#indxlagparams=paramsList.index(lagparamsList[0])
testensemble=ensemble[:,1,:]
iparams=newmcmc.tr.all_parameters
lagt=np.zeros((nmodels*newmcmc.nwalkers,len(times)))
acct=np.zeros((nmodels*newmcmc.nwalkers,len(obl_times)))
tmpt=np.zeros((nmodels*newmcmc.nwalkers,len(obl_times),2))

indxw=0

for w in range(0,newmcmc.nwalkers):
    for i in range(0,nmodels):
        iparams=dict(zip(newmcmc.tr.all_parameter_names,ensemble[i,w,:]))
        newmcmc.tr.set_model(iparams)
        
        lagti=newmcmc.tr.lagModel.get_lag_at_t(times)
        accti=newmcmc.tr.accuModel.get_accumulation_at_t(obl_times)
        tmpti=np.array(newmcmc.tr.get_trajectory(obl_times))
        
        lagt[indxw]=lagti
        acct[indxw]=accti
        tmpt[indxw,:,:]=tmpti.T
        indxw=indxw+1
        
        
plt.figure()

#plot lagt
plt.subplot(4,1,1)
plt.plot(times,lagt.T)
plt.xticks([], [])
plt.title('Lag (mm)')

#plot lagt
plt.subplot(4,1,2)
plt.plot(obl_times,acct.T)
plt.xticks([], [])
plt.title('acc rate (m/year)')

#plot yt
plt.subplot(4,1,3)
plt.plot(obl_times,tmpt[:,:,1].T)
plt.title('Vertical distance (m)')
plt.xticks([], [])

#plot xt
plt.subplot(4,1,4)
plt.plot(obl_times,tmpt[:,:,0].T)
plt.xlabel('Time (years)')
plt.title('Horizontal distance (m)')

#create folder for saving figure
if not os.path.exists(newmcmc.directory+'figures/'+'lagaccdist/'):
    os.makedirs(newmcmc.directory+'figures/'+'lagaccdist/')
    
plt.savefig(newmcmc.directory+'figures/'+'lagaccdist/'
            +newmcmc.modelName+'_'+str(newmcmc.maxSteps)+'.pdf',
            facecolor='w',pad_inches=0.1)

# tmp for opt params, params with highest log prob and 100 random models in ensemble
#reshape logprob
plt.figure()

logprob2d=logprob.T.reshape(1,nmodels*newmcmc.nwalkers)
#best model params
bestTMPindx=np.argmax(logprob2d)
bestTMP=tmpt[bestTMPindx,:,:]
plt.plot(bestTMP[:,0],bestTMP[:,1],c='b',label='Best TMP')

ratioyx=0.2;

#find nearest points
ndata=len(xdata)
x_model=bestTMP[:,0]
y_model=bestTMP[:,1]
xnear = np.zeros_like(xdata)
ynear = np.zeros_like(ydata)
timenear = np.zeros_like(xdata)

for i, (xdi, ydi) in enumerate(zip(xdata, ydata)):
    dist = newmcmc.tr._L2_distance(x_model, xdi, y_model, ydi)
    ind = np.argmin(dist)
    xnear[i] = x_model[ind]
    ynear[i] = y_model[ind]
    timenear[i] = obl_times[ind]
    
#plot tmp data assuming errorbar is last errorbar of mcmc
xerr, yerr = newmcmc.tr.errorbar*newmcmc.tr.meters_per_pixel
plt.errorbar(x=xdata, xerr=xerr, y=ydata, yerr=yerr, 
             c='r', marker='.', ls='',label='Observed TMP')

for i in range(nmodels):
    indx=np.random.randint(0,nmodels*newmcmc.nwalkers)
    plt.plot(tmpt[indx,:,0],tmpt[indx,:,1],c="gray", alpha=0.1, zorder=-1)
plt.plot(tmpt[indx,:,0],tmpt[indx,:,1],c="gray", alpha=0.1, zorder=-1,label='Ensemble models')
plt.xlabel("Horizontal dist [m]")
plt.ylabel("V. dist [m]")
ax=plt.gca()
ax.legend()
ymin,ymax=ax.get_ylim()
xmin,xmax=ax.get_xlim()
ax.set_ylim(ymin,0)
ax.set_xlim(0,xmax)
ax.set_box_aspect(ratioyx)

#plot times on right axis
ax2=ax.twiny()
color='m'
ax2.set_xlabel('Time before present ( 10^3 years)',color=color)
plt.scatter(xnear,ynear,marker="o",color='m')
ax2.set_ylim(ymin,0)
ax2.set_xlim(0,xmax)
ax2.tick_params(axis='x',labelcolor=color)
plt.xticks(xnear,np.round(timenear/1000).astype(int),rotation=90)
ax2.set_box_aspect(ratioyx)

#create folder for saving figure
if not os.path.exists(newmcmc.directory+'figures/'+'tmp/'):
    os.makedirs(newmcmc.directory+'figures/'+'tmp/')
    
plt.savefig(newmcmc.directory+'figures/'+'tmp/'
            +newmcmc.modelName+'_'+str(newmcmc.maxSteps)+'.pdf',
            facecolor='w',pad_inches=0.1)

