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
        thin_by: int,
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
        self.thin_by = thin_by
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
                       lag_params,tmp,errorbar,angle)
        
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
        #self.optParams=guessParams
        
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
        self.filename=self.directory+'obj/'+self.modelName+'_'+str(self.maxSteps)+'obj'
        
        #Set optimized parameter values as initial values of MCMC chains 
        self.initParams=np.array([self.optParams+ 
                        1e-3*self.optParams*np.random.randn(self.ndim) 
                        for i in range(self.nwalkers)])
        
        #check that all initParams fit the priors
        likeInit=np.zeros((self.nwalkers,1))
        for i in range(0,self.nwalkers):
            iparams=dict(zip(self.tr.all_parameter_names,self.initParams[i,:]))
            likeInit[i]=self.ln_likelihood(iparams)
        
        indxProblem=[i for i, n in enumerate(likeInit) if n == -1e99]
        
        if len(indxProblem)>0:
            for i in range(0,len(indxProblem)):
                like=-1e99
                while like==-1e99:
                    iparamsTest=(self.optParams+ 1e-3*self.optParams*
                                 np.random.randn(self.ndim))
                    iparamsDict=dict(zip(self.tr.all_parameter_names,iparamsTest))
                    like=self.ln_likelihood(iparamsDict)
                self.initParams[indxProblem[i],:]=iparamsTest
                
        sys.stderr.flush()  
        print('All initial parameters fit priors',file=sys.stderr)
        sys.stderr.flush()

        start = time.time()
        
        #Initialize sampler
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, 
                                             self.ln_likelihood,  
                                             parameter_names=self.parameter_names)
        #Run MCMC and track progress
        self.sampler.reset()
        #Iteratively compute autocorrelation time Tau
        index=0
        self.autocorr=np.zeros(10)
        self.taus=np.zeros((10,self.ndim))
        old_tau=np.inf
        
        #compute tau every n iterations
        for sample in self.sampler.sample(self.initParams,iterations=self.maxSteps, 
        progress=False):
        
            if self.sampler.iteration%(self.maxSteps/10):
                continue
            sys.stderr.flush()
            print(self.sampler.iteration/self.maxSteps*100,'%',file=sys.stderr)
            
            tau=self.sampler.get_autocorr_time(tol=0)
            self.autocorr[index]=np.mean(tau)
            self.taus[index,:]=tau
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
        
        
        #save samples and log prob every thin_by steps
        all_samples=self.sampler.get_chain()
        self.samples=all_samples[0::self.thin_by,:,:]
        all_logprob=self.sampler.get_log_prob()
        self.logprob=all_logprob[0::self.thin_by,:]
        self.accFraction=self.sampler.acceptance_fraction
        self.totalSteps=self.sampler.iteration
        
        #delete sampler because it is very large
        del self.sampler
        

    def ln_likelihood(self,params: Dict[str,float]):
        
        errorbar: float = params["errorbar"]
        
        if errorbar < 0: #prior on the variance (i.e. the error bars)
            return -1e99
        
        #set trough model with candidate parameters
        self.tr.set_model(params)
        #compute likelihood of model 
        likelihood_of_model=self.tr.lnlikelihood(self.xdata,self.ydata,
                                                 self.tr.accuModel._times)
        #prior lag with time
        if any(self.tr.lag_at_t  < 1e-15) or any(self.tr.lag_at_t > 20):
            return -1e99
        
        #prior nearest points to observed data
        if any(self.tr.ynear < -2e3) or any(self.tr.ynear > 0):
            return -1e99
        
        #get accumulation with time
        acc_t=self.tr.accuModel.get_accumulation_at_t(self.tr.accuModel._times)
        if any(acc_t < 0):
            return -1e99
        
        #get exponent of accumulation, if it exists
        if "acc_exponent" in params.keys():
            exponent: float = params["acc_exponent"]
            
            if exponent < -3:
                return -1e99
        
        return likelihood_of_model
    
    #And the negative of the log likelihood
    def neg_ln_likelihood(self,paramsArray):
        
        params=dict(zip(self.parameter_names, paramsArray))
        
        return -self.ln_likelihood(params)

    



        

        