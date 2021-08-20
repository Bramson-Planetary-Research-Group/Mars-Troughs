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

listObj_ins1=glob.glob("../../outputMCMC/insolation/TMP1/obj/*obj")
listObj_ins2=glob.glob("../../outputMCMC/insolation/TMP2/obj/*obj")
listObj_obl1=glob.glob("../../outputMCMC/obliquity/TMP1/obj/*obj")
listObj_obl2=glob.glob("../../outputMCMC/obliquity/TMP2/obj/*obj")
listObj=listObj_ins1+listObj_ins2+listObj_obl1+listObj_obl2

numfiles=len(listObj)

BICs=np.zeros(numfiles)

for i in np.arange(0,numfiles):

    infile=open(listObj[i],'rb')
    newmcmc=pickle.load(infile)
    infile.close()
    
    nmodels=100
    totalsteps=newmcmc.sampler.iteration
    initmodel=totalsteps-nmodels
    
    #get log prob
    all_logprob=newmcmc.sampler.get_log_prob()
    logprob=all_logprob[initmodel:,:]
    logprob2d=logprob.T.reshape(1,nmodels*newmcmc.nwalkers)
    #best model params
    bestmodel=np.argmax(logprob2d)
    
    #get number of parameters
    paramsList=list(newmcmc.tr.all_parameter_names)
    numparams=len(paramsList)
    
    #get number of tmp data points
    n=len(newmcmc.ydata)
    
    #compute BIC
    BIC=logprob2d[0,bestmodel]-1/2*numparams*np.log(n)
    BICs[i]=BIC
    
maxBIC=np.argmax(BICs)
    
plt.figure
plt.plot(BICs)
plt.xlabel('Model')
plt.ylabel('BIC')
plt.title(listObj[maxBIC])
plt.show()