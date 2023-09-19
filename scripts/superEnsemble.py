#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 18:48:32 2022

@author: kris
"""

import numpy as np
import pickle

def main():
    
    tmp='all'
    
    obj='/Users/kris/Documents/work/purdue/MarsTroughsProject/outputMCMC/manytmps/trough1/TMP1/obliquity/obj/Quadratic_Obliquity_PowerLawLag_100000obj'
        
    #load obj
    infile=open(obj,'rb')
    mcmc=pickle.load(infile)
    infile.close()
    
    all_ages,all_accts,all_lagt,all_retreats,all_xts,all_yts,all_logprobs= reshapeEnsemble(mcmc)

        
    timeaxis=mcmc.tr.accuModel._times/1000000

    times=[]
    retreatsplot=[]
    accsplot=[]
    lagsplot=[]
    xsplot=[]
    ysplot=[]
    
    nmodelsSuper=len(all_ages)
    print(nmodelsSuper)
    
    for i in range(0,nmodelsSuper):
        
        timesi=timeaxis[timeaxis<all_ages[i]]
        times=np.concatenate((times,timesi),axis=0)
        
        retreatsi=all_retreats[i,timeaxis<all_ages[i]]
        retreatsplot=np.concatenate((retreatsplot,retreatsi),axis=0)
        
        acci=all_accts[i,timeaxis<all_ages[i]]
        accsplot=np.concatenate((accsplot,acci),axis=0)
        
        lagi=all_lagt[i,timeaxis<all_ages[i]]
        lagsplot=np.concatenate((lagsplot,lagi),axis=0)
        
        xti=all_xts[i,timeaxis<all_ages[i]]
        yti=all_yts[i,timeaxis<all_ages[i]]
        xsplot=np.concatenate((xsplot,xti),axis=0)
        ysplot=np.concatenate((ysplot,yti),axis=0)
        
        print(i/nmodelsSuper*100)
        
    superEnsemble=Empty()
    superEnsemble.ages=all_ages
    superEnsemble.accts=all_accts
    superEnsemble.lagts=all_lagt
    superEnsemble.retreats=all_retreats
    superEnsemble.xts=all_xts
    superEnsemble.yts=all_yts
    superEnsemble.logprobs=all_logprobs
    superEnsemble.mcmcobj=mcmc
    superEnsemble.times=times
    superEnsemble.retreatsplot=retreatsplot
    superEnsemble.accsplot=accsplot
    superEnsemble.lagsplot=lagsplot
    superEnsemble.xsplot=xsplot
    superEnsemble.ysplot=ysplot
    superEnsemble.tmp=tmp

    
    filename='../../outputMCMC/manytmps/trough1/TMP1/obliquity/obj/superEnsemble_' + str(tmp)
    outfile=open(filename,'wb')
    pickle.dump(superEnsemble,outfile)
    outfile.close()
    
def reshapeEnsemble(newmcmc):
    
    lastnmodels=100
    stepEnsemble=1
    
    nchains=newmcmc.samples.shape[1]

    #subsample ensemble
    ensemble=newmcmc.samples[-1*lastnmodels::stepEnsemble,:,:]
    logprob=newmcmc.logprob[-1*lastnmodels::stepEnsemble,:]
    xaxis=np.arange(newmcmc.totalSteps-(lastnmodels-1)*newmcmc.thin_by,newmcmc.totalSteps+1,stepEnsemble*newmcmc.thin_by)
    nmodels=len(xaxis) 
    logprob1d=logprob.reshape(nmodels*nchains,1)
    
    #lag, acc rate and y per time for each model ----------------------------
        #indxlagparams=paramsList.index(lagparamsList[0])
        
    retreatt=np.zeros((nmodels*nchains,len(newmcmc.tr.accuModel._times)))
    lagt=np.zeros((nmodels*nchains,len(newmcmc.tr.accuModel._times)))
    acct=np.zeros((nmodels*nchains,len(newmcmc.tr.accuModel._times)))
    yt=np.zeros((nmodels*nchains,len(newmcmc.tr.accuModel._times)))
    xt=np.zeros((nmodels*nchains,len(newmcmc.tr.accuModel._times)))
    
    indxw=0
    for i in range(0,nmodels):
        for w in range(0,nchains):
            iparams=dict(zip(newmcmc.tr.all_parameter_names,ensemble[i,w,:]))
            newmcmc.tr.set_model(iparams)
            
            retreati=newmcmc.tr.retreat_model_t
            lagti=newmcmc.tr.lagModel.get_lag_at_t(newmcmc.tr.accuModel._times)
            accti=newmcmc.tr.accuModel.get_accumulation_at_t(newmcmc.tr.accuModel._times)
            tmpti=np.array(newmcmc.tr.get_trajectory(newmcmc.tr.accuModel._times))
            
            retreati[retreati<0]=0
            retreatt[indxw]=retreati*1000
            lagt[indxw]=lagti
            acct[indxw]=accti*1000
            xt[indxw,:]=tmpti[0,:]
            yt[indxw,:]=tmpti[1,:]
            indxw=indxw+1
            
    ndata=len(newmcmc.xdata[0])
    lastxdata=newmcmc.xdata[0][ndata-1]
    lastydata=newmcmc.ydata[0][ndata-1]
    ages = np.zeros((nmodels*nchains,1))
    
    for w in range(0,nmodels*nchains):
        xi=xt[w]
        yi=yt[w]
        disti = newmcmc.tr._L2_distance(xi, lastxdata, yi, lastydata)
        ind = np.argmin(disti)
        ages[w] = newmcmc.tr.accuModel._times[ind]/1000000
        
    return ages,acct,lagt,retreatt,xt,yt,logprob1d

class Empty:
    pass