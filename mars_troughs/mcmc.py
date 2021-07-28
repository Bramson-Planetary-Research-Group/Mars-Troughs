#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 09:31:34 2021

@author: kris
"""
#import modules
import time
from typing import Dict, List, Optional, Union
import numpy as np
import scipy.optimize as op
import mars_troughs as mt
import emcee
from mars_troughs import DATAPATHS, Model
import os

class MCMC():
    """
    Class for running MCMC chains to obtain a sample of climate parameters
    of trough migration.
    """
    def __init__(
        self,
        maxSteps: int,
        subIter: int,
        directory: str,
        acc_model = Union[str, Model],
        lag_model = Union[str, Model],
        acc_params: Optional[List[float]] = None,
        lag_params: Optional[List[float]] = None,
        errorbar = np.sqrt(1.6), #errorbar in pixels on the datapoints
        angle= 5.0,
    ):
        self.maxSteps = maxSteps
        self.subIter = subIter
        self.acc_model = acc_model
        self.lag_model = lag_model
        
        #Load data
        self.xdata,self.ydata=np.loadtxt(DATAPATHS.TMP, unpack=True) #Observed TMP data
        self.xdata=self.xdata*1000 #km to m 
        
        # Create  trough object 
        self.tr = mt.Trough(self.acc_model,self.lag_model,acc_params,
                       lag_params,
                       errorbar,angle)
        
        self.parameter_names = ([key for key in self.tr.all_parameters])
        
        #Find number of dimensions and number of parameters per submodel
        self.ndim=len(self.parameter_names)
        self.nwalkers=self.ndim*4
        
        
        #Linear optimization
        
        guessParams=np.array([errorbar]
                             +list(self.tr.accuModel.parameters.values())
                             +list(self.tr.lagModel.parameters.values()))
        optObj= op.minimize(self.neg_ln_likelihood, x0=guessParams, 
                            method='Nelder-Mead')
        self.optParams=optObj['x']
        
        
        #Create directory to save ensemble and figures
        if isinstance(self.acc_model, str):
            #do nothing
            acc_model_name=self.acc_model
        else:
            auxAcc=str(self.acc_model).split(' ')
            auxAcc=auxAcc[0]
            acc_model_name=auxAcc.split('.')
            acc_model_name=acc_model_name[2]
            
        
        if isinstance(self.lag_model, str):
            #do nothing
            lag_model_name=self.lag_model
        else:
            auxLag=str(lag_model).split(' ')
            auxLag=auxLag[0]
            lag_model_name=auxLag.split('.')
            lag_model_name=lag_model_name[2]
        
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.subdir=acc_model_name+'_'+lag_model_name+'/'
        if not os.path.exists(directory+self.subdir):
            os.makedirs(directory+self.subdir)
    
        self.filename=directory+self.subdir+str(self.maxSteps)
        
        #Set file to save progress 
        backend=emcee.backends.HDFBackend(self.filename+'.h5')
        backend.reset(self.nwalkers,self.ndim)
        
        #Set optimized parameter values as initial values of MCMC chains 
        self.initParams=np.array([self.optParams+ 
                        1e-3*self.optParams*np.random.randn(self.ndim) 
                        for i in range(self.nwalkers)])
    
        start = time.time()
        
        #Initialize sampler
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, 
                                             self.ln_likelihood, 
                                             backend=backend, 
                                             parameter_names=self.parameter_names)
        #Run MCMC and track progress
        self.sampler.reset()
        #Iteratively compute autocorrelation time Tau
        index=0
        autocorr=np.zeros(int(self.maxSteps/self.subIter))
        old_tau=np.inf
        
        #compute tau every subIter iterations
        for sample in self.sampler.sample(self.initParams,iterations=self.maxSteps, 
        progress=False):
        
            if self.sampler.iteration%self.subIter:
                continue
                
            print(self.sampler.iteration/self.maxSteps*100,'%')
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

        print("Running time {0:.1f} seconds".format(running_time), 
              'for',self.subdir,self.maxSteps)

    def ln_likelihood(self,params: Dict[str,float]):
        
        errorbar: float = params["errorbar"]
        
        if errorbar < 0: #prior on the variance (i.e. the error bars)
            return -1e99
    
        self.tr.set_model(params)
        
        lag_t=self.tr.lagModel.get_lag_at_t(self.tr.times)
        
    
        if any(lag_t < 1e-10) or any(lag_t > 20):
            
            return -1e99
    
        return self.tr.lnlikelihood(self.xdata,self.ydata)
    
    #And the negative of the log likelihood
    def neg_ln_likelihood(self,paramsArray):
        
        params=dict(zip(self.parameter_names, paramsArray))
        
        return -self.ln_likelihood(params)

    



        

        