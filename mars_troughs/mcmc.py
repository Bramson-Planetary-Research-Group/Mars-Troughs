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
import sys

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
        tmp: int,
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
        self.directory = directory
        self.tmp=tmp
        
        #Load data
        if tmp==1:
            self.xdata,self.ydata=np.loadtxt(DATAPATHS.TMP1, 
                                         unpack=True) #Observed TMP data
        else:
            self.xdata,self.ydata=np.loadtxt(DATAPATHS.TMP2, 
                                         unpack=True) #Observed TMP data    
        
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
        
        #aux for creating directories
        
        if isinstance(self.acc_model, str):
            #do nothing
            self.acc_model_name=self.acc_model
        else:
            auxAcc=str(self.acc_model).split(' ')
            auxAcc=auxAcc[0]
            self.acc_model_name=auxAcc.split('.')
            self.acc_model_name=self.acc_model_name[2]
            
        
        if isinstance(self.lag_model, str):
            #do nothing
            self.lag_model_name=self.lag_model
        else:
            auxLag=str(self.lag_model).split(' ')
            auxLag=auxLag[0]
            self.lag_model_name=auxLag.split('.')
            self.lag_model_name=self.lag_model_name[2]
        
        #Create directory to save outputs
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        #Create directory to save objects and h5 tracking file
        if not os.path.exists(self.directory+'obj/'):
            os.makedirs(self.directory+'obj/')
            
        self.modelName=self.acc_model_name+'_'+self.lag_model_name
        #if not os.path.exists(self.directory+'obj/'+self.modelName+'/'):
         #   os.makedirs(self.directory+'obj/'+self.modelName+'/')
    
        #self.filename=self.directory+'obj/'+self.modelName+'/'+str(self.maxSteps)
        self.filename=self.directory+'obj/'+self.modelName+'_'+str(self.maxSteps)
        
        #Set file to save progress 
        backend=emcee.backends.HDFBackend(self.filename+'.h5')
        backend.reset(self.nwalkers,self.ndim)
        
        #Set optimized parameter values as initial values of MCMC chains 
        self.initParams=np.array([self.optParams+ 
                        1e-3*self.optParams*np.random.randn(self.ndim) 
                        for i in range(self.nwalkers)])
        
        print(self.initParams,file=sys.stderr)
    
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
        self.autocorr=np.zeros(int(self.maxSteps/self.subIter))
        old_tau=np.inf
        
        #compute tau every subIter iterations
        for sample in self.sampler.sample(self.initParams,iterations=self.maxSteps, 
        progress=False):
        
            if self.sampler.iteration%self.subIter:
                continue
                
            print(self.sampler.iteration/self.maxSteps*100,'%',file=sys.stderr)
            sys.stderr.flush()
            
            tau=self.sampler.get_autocorr_time(tol=0)
            self.autocorr[index]=np.mean(tau)
            index+=1
            converged=np.all(tau*50<self.sampler.iteration)
            converged&=np.all(np.abs(old_tau-tau)/tau<0.01)
            
            if converged:
                break
            
            old_tau=tau
            
        end = time.time()
        running_time=end-start

        print("Running time {0:.1f} seconds".format(running_time), 
              'for',self.modelName,self.maxSteps,file=sys.stderr)
        sys.stderr.flush()

    def ln_likelihood(self,params: Dict[str,float]):
        
        errorbar: float = params["errorbar"]
        
        if errorbar < 0: #prior on the variance (i.e. the error bars)
            return -1e99
    
        self.tr.set_model(params)
        
        lag_t=self.tr.lagModel.get_lag_at_t(self.tr.accuModel._times)
    
        if any(lag_t < 1e-15) or any(lag_t > 20):
            return -1e99
    
        y = self.tr.accuModel.get_yt(self.tr.accuModel._times)
        
        if any(y < -2e3) or any(y > 0):
            return -1e99
        
        acc_t=self.tr.accuModel.get_accumulation_at_t(self.tr.accuModel._times)
        
        if any(acc_t < 0):
            return -1e99
        
        return self.tr.lnlikelihood(self.xdata,self.ydata)
    
    #And the negative of the log likelihood
    def neg_ln_likelihood(self,paramsArray):
        
        params=dict(zip(self.parameter_names, paramsArray))
        
        return -self.ln_likelihood(params)

    



        

        