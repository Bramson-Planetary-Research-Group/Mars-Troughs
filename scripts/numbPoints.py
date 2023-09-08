#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 13:23:47 2023

@author: kris
"""
#create subsamples of TMP 1 data points, use each subsample as a different
#TMP in the tmp inversion code, and compare the inferred accumulation rates 
#and age of each subsample 
from optimization import Optimization
from mars_troughs import DATAPATHS
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from mars_troughs import (load_insolation_data)
from mars_troughs import PowerLawLag
from mars_troughs import Quadratic_Insolation




def main():

    #load original TMP 1 and 2 data
    xdata1,ydata1 = np.loadtxt(DATAPATHS.TMP1, 
                              unpack=True) #Observed TMP 1 data
    xdata1=xdata1*1000 #km to m
    N=len(xdata1)
    
    xdata2,ydata2 = np.loadtxt(DATAPATHS.TMP2, 
                              unpack=True) #Observed TMP 1 data
    xdata2=xdata2*1000 #km to m
    n=len(xdata2)
    
    #uncertainty data 
    stdx=475 #m
    stdy=20 #m
    
    #get ns subsamples of original tmp 1 data 
    #ns is found by command combinations
    #keep first and last TMP 1 points in all subsamples (use n-2)
    indexAll=np.arange(1,N-1)
    subsamples=list(combinations(indexAll,n-2)) 
    nsamples=len(subsamples)
    #get subsampled TMPs
    subx=np.zeros((nsamples,n),dtype=float)
    suby=np.zeros((nsamples,n),dtype=float)

    for i in range(0,nsamples):
        subxAux=xdata1[0]
        subsamplex=xdata1[np.array(subsamples[i])]
        subxAux=np.append(subxAux,subsamplex)
        subxAux=np.append(subxAux,xdata1[-1])
        subx[i]=subxAux
        
        subyAux=ydata1[0]
        subsampley=ydata1[np.array(subsamples[i])]
        subyAux=np.append(subyAux,subsampley)
        subyAux=np.append(subyAux,ydata1[-1])
        suby[i]=subyAux
        
    #plot subsamples TMPs
    plt.figure()
    plt.plot(subx.T,suby.T)
    plt.errorbar(x=xdata1, 
                 xerr=stdx,
                 y=ydata1, 
                 yerr=stdy, 
                 c='r', marker='.', ls='',label='Observed TMP 1')
    plt.xlabel("Horizontal dist [km]")
    plt.ylabel("V. dist [m]")
    plt.legend()
    ratioyx=0.4;
    ax=plt.gca()
    ax.set_box_aspect(ratioyx)
    plt.show()
    plt.savefig('../../outputSubs/allSubsTMPs.pdf')
    
    #obtain mean accumulation rate and age of trough from each subsampled tmp
    tmp=1
    angle=2.9
    (insolations,times) = load_insolation_data(tmp)
    times=-times.astype(float)
    times[0]=1e-10
    acc_model=Quadratic_Insolation(times,insolations)
    lag_model=PowerLawLag()
    
    #loop through each subsample and plot
    meanAccs=np.zeros(nsamples)
    ages=np.zeros(nsamples)
    chis=np.zeros(nsamples)
    
    plt.figure()
    opt = Optimization(tmp,angle,acc_model,lag_model)
    
    for i in range(0,nsamples):
        #init optimiztion class
        
        # plt.errorbar(x=subx[i],
        #              xerr=stdx,
        #              y=suby[i], 
        #              yerr=stdy, 
        #              c='r', marker='.', ls='',label='Subsample TMP 1')
        
        (xinit,yinit,optParams, xmodel, ymodel, 
         meanAcc, age, chi2sum) = opt.optFitSubTMP(subx[i],suby[i])
        
        meanAccs[i]=meanAcc
        ages[i]=age
        chis[i]=chi2sum
        
        plt.plot(xmodel,ymodel,c='b',alpha=0.5)
        #plt.plot(xinit,yinit,c='g',label='Default TMP')
        print(i)
    ax=plt.gca()
    ax.set_box_aspect(ratioyx)
    plt.show()
    plt.savefig('../../outputSubs/modelTmpsSubsamples.pdf')
    
    plt.figure()
    nbins=10
    plt.hist(meanAccs,nbins)
    plt.xlabel('mean accumulation  (mm/y)')
    plt.ylabel('Number of subsamples')
    plt.savefig('../../outputSubs/histMeanAcc.pdf')
    
    plt.figure()
    plt.hist(ages/1e6,nbins)
    plt.xlabel('ages  (Myr)')
    plt.ylabel('Number of subsamples')
    plt.savefig('../../outputSubs/ages.pdf')
    
    plt.figure()
    plt.hist(chis,nbins)
    plt.xlabel('residual  (m)')
    plt.ylabel('Number of subsamples')
    plt.savefig('../../outputSubs/chi.pdf')
    
    
    return meanAccs,ages,chis
    
