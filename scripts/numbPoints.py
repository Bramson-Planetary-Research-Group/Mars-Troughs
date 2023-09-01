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
import mars_troughs as mt
from mars_troughs import (DATAPATHS, Model, load_retreat_data)
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from mars_troughs import (load_insolation_data, 
                          load_obliquity_data)
from mars_troughs import PowerLawLag
from mars_troughs import Quadratic_Insolation
from typing import Dict, List, Optional, Union



def main():

    #load original TMP 1 and 2 data
    xdata1,ydata1 = np.loadtxt(DATAPATHS.TMP1, 
                              unpack=True) #Observed TMP 1 data
    N=len(xdata1)
    
    xdata2,ydata2 = np.loadtxt(DATAPATHS.TMP2, 
                              unpack=True) #Observed TMP 1 data
    n=len(xdata2)
    
    #uncertainty data 
    stdx=0.475 #km
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
    
    #obtain mean accumulation rate and age of trough from each subsampled tmp
    
    (insolations,times) = load_insolation_data(1)
    times=-times.astype(float)
    times[0]=1e-10
    acc_model=Quadratic_Insolation(times,insolations)
    lag_model=PowerLawLag()
    
    for i in range(0,2):
        optParams, xmodel, ymodel, meanAcc, age, chi2 = optimization(
            subx[i],suby[i],acc_model,lag_model,1)
    
    
