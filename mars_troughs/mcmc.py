#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 09:31:34 2021

@author: kris
"""
#import modules
import os
import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import mars_troughs as mt
import emcee, corner
from importlib import reload
from mars_troughs import DATAPATHS, Model
from mars_troughs.datapaths import (
    load_insolation_data,
    load_obliquity_data,
    load_retreat_data)

class MCMC():
    """
    Class for running MCMC chains to obtain a sample of climate parameters
    of trough migration.
    """
    def __init__(
        self,
        maxSteps: int,
        subIter: int,
        filename: str,
        acc_model_name = Union[str, Model],
        lag_model_name = Union[str, Model],
        acc_params: Optional[List[float]] = None,
        lag_params: Optional[List[float]] = None,
        errorbar = np.sqrt(1.6), #errorbar in pixels on the datapoints
        angle= 5.0,
    ):
        self.maxSteps = maxSteps
        self.subIter = subIter
        self.filename = filename
        self.acc_model_name = acc_model_name
        self.lag_model_name = lag_model_name
        
        # Create  trough object 
        self.tr = mt.Trough(self.acc_model_name,self.lag_model_name,acc_params,
                       lag_params,
                       errorbar,angle)
        breakpoint()
        
        self.parameter_names = ([key for key in self.tr.all_parameters])
        
        #Find number of dimensions and number of parameters per submodel
        self.ndim=len(self.parameter_names)
        self.nwalkers=ndim*4
        
        #Define the log likelihood
    
    #Linear optimization
        guessParams=np.array([errorbar]+acc_params+lag_params)
        optObj= op.minimize(self.neg_ln_likelihood, x0=guessParams, 
                            method='Nelder-Mead')
        self.optParams=optObj['x']
        
        #Set file to save progress 
        backend=emcee.backends.HDFBackend(filename+'.h5')
        backend.reset(nwalkers,ndim)
        
        #Set optimized parameter values as initial values of MCMC chains 
        self.initParams=np.array([optParams+ 
                        1e-3*optParams*np.random.randn(ndim) 
                        for i in range(nwalkers)])
    
        start = time.time()
        
        #Initialize sampler
        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.ln_likelihood, 
                                        backend=backend, parameter_names=self.parameter_names)
        #Run MCMC and track progress
        self.sampler.reset()
        #Iteratively compute autocorrelation time Tau
        index=0
        autocorr=np.zeros(int(maxSteps/subIter))
        old_tau=np.inf
        
        #compute tau every subIter iterations
        for sample in self.sampler.sample(self.initParams,iterations=self.maxSteps, 
        progress=True):
        
            if self.sampler.iteration%self.subIter:
                continue
                
            tau=self.sampler.get_autocorr_time(tol=0)
            autocorr[index]=np.mean(tau)
            index+=1
            converged=np.all(tau*50<self.sampler.iteration)
            converged&=np.all(np.abs(old_tau-tau)/tau<0.01)
            
            if converged:
                break
            
            old_tau=tau
            
        end = time.time()
        running_time=end-start

        print("MCMC running time {0:.1f} seconds".format(running_time))

    def ln_likelihood(self,params: Dict[str,float]):
        
        errorbar: float = params["errorbar"]
        
        if errorbar < 0: #prior on the variance (i.e. the error bars)
            return -1e99
    
        self.tr.set_model(params)
        
        lag_t=self.tr.lagModel.get_lag_at_t(times)
    
        if any(lag_t < 0) or any(lag_t > 20):
    
            return -1e99
    
        return self.tr.lnlikelihood(xdata,ydata)
    
    #And the negative of the log likelihood
    def neg_ln_likelihood(self,paramsArray):
        
        params=dict(zip(self.parameter_names, paramsArray))
        
        return -ln_likelihood(params)

    



        

        