#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 10:32:35 2021

@author: kris
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
import glob
import os
import argparse

p=argparse.ArgumentParser(description="TMP data")
p.add_argument("-tmp",type=int,help="TMP 1 or 2")
p.add_argument("-initmodel",type=int,help="initial model for plotting")
p.add_argument("-stepEnsemble",type=int,help="skip models for plotting")
args=p.parse_args()

listObj_ins1=glob.glob("../../outputMCMC/insolation/TMP1/obj/*obj*")
listObj_ins1.sort()
listObj_ins2=glob.glob("../../outputMCMC/insolation/TMP2/obj/*obj*")
listObj_ins2.sort()
listObj_obl1=glob.glob("../../outputMCMC/obliquity/TMP1/obj/*obj*")
listObj_obl1.sort()
listObj_obl2=glob.glob("../../outputMCMC/obliquity/TMP2/obj/*obj*")
listObj_obl2.sort()

if args.tmp==1:
    listObj=listObj_ins1+listObj_obl1
else:
    listObj=listObj_ins2+listObj_obl2

numfiles=len(listObj)

BICs=np.zeros(numfiles)
ages=np.zeros(numfiles)
modelNames=['Models']

for i in np.arange(0,numfiles):

    infile=open(listObj[i],'rb')
    newmcmc=pickle.load(infile)
    infile.close()
    #get number of parameters
    numparams=len(newmcmc.parameter_names)
    #get number of tmp data points
    n=len(newmcmc.ydata)
    
    #subsample ensemble
    ensemble=newmcmc.samples[int(args.initmodel/newmcmc.thin_by-1
                                 )::args.stepEnsemble,:,:]
    xaxis=np.arange(args.initmodel,newmcmc.totalSteps+1,
                    args.stepEnsemble*newmcmc.thin_by)
    nmodels=len(xaxis) 
    logprob=newmcmc.logprob[int(args.initmodel/newmcmc.thin_by-1
                                )::args.stepEnsemble,:]
    
    #get tmps
    tmpt=np.zeros((nmodels*newmcmc.nwalkers,
                   len(newmcmc.tr.accuModel._times),2))
        
    indxw=0
    for i in range(0,nmodels):
        for w in range(0,newmcmc.nwalkers):
            iparams=dict(zip(newmcmc.tr.all_parameter_names,ensemble[i,w,:]))
            newmcmc.tr.set_model(iparams)
            tmpti=np.array(newmcmc.tr.get_trajectory(newmcmc.tr.accuModel._times))
            tmpt[indxw,:,:]=tmpti.T
            indxw=indxw+1
    #get corresponding likelihoods
    logprob1d=logprob.reshape(nmodels*newmcmc.nwalkers,1)
    indxbest=np.argmax(logprob1d)
    
    #find ages
    lastxdata=newmcmc.xdata[n-1]
    lastydata=newmcmc.ydata[n-1]
    agesModel = np.zeros((nmodels*newmcmc.nwalkers,1))

    for w in range(0,nmodels*newmcmc.nwalkers):
        xi=tmpt[w,:,0]
        yi=tmpt[w,:,1]
        disti = newmcmc.tr._L2_distance(xi, lastxdata, yi, lastydata)
        ind = np.argmin(disti)
        agesModel[w] = newmcmc.tr.accuModel._times[ind]/1000000
    

    #compute BIC
    BIC=logprob1d[indxbest]-1/2*numparams*np.log(n)
    age=agesModel[indxbest]
    BICs[i]=BIC
    ages[i]=age
    modelNames=modelNames+[newmcmc.modelName]
    
    
maxBIC=np.argmax(BICs)
bestAge=ages[maxBIC]
    
plt.figure
plt.plot(ages,BICs)
plt.xlabel('ages')
plt.ylabel('BIC')
plt.title(modelNames[maxBIC]+'_'+str(bestAge)+' Myr')
plt.subplots_adjust(bottom=0.6)


plt.savefig('../../outputMCMC/BICs/TMP' +str(args.tmp)+'age.pdf',
            facecolor='w',pad_inches=0.1)

#save bic obj

outfile=open('../../outputMCMC/BICs/objTMP'+str(args.tmp)+'age','wb')
pickle.dump(BICs,outfile)
pickle.dump(modelNames,outfile)
outfile.close()