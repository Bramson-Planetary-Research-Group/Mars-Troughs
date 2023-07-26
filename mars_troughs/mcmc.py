#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 09:31:34 2021

@author: kris
@edit: kris laferriere, update to use retreat, not lag. 
"""
#import modules
import time
from typing import Dict, List, Optional, Union
import numpy as np
import scipy.optimize as op
import mars_troughs as mt
import emcee
from mars_troughs import (DATAPATHS, Model)
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
        retr_model = Union[str, Model],
        #errorbar = np.sqrt(1.6), #errorbar in pixels on the datapoints
        angle= 5.0,
    ):
        self.maxSteps = maxSteps
        self.thin_by = thin_by
        self.acc_model = acc_model
        self.retr_model = retr_model
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
        self.tr = mt.Trough(self.acc_model,self.retr_model, angle)
                            #ret_data_spline, 
                            #errorbar,angle)

                            #ret_data_spline,errorbar,angle)
        
        self.parameter_names = ([key for key in self.tr.all_parameters])
        
        #Find number of dimensions and number of parameters per submodel
        self.ndim=len(self.parameter_names)
        self.nwalkers=self.ndim*4
        
        
        #Linear optimization
        
        guessParams=np.array(#[errorbar]+
                             list(self.tr.accuModel.parameters.values())
                             +list(self.tr.retrModel.parameters.values()))
        optObj= op.minimize(self.neg_ln_likelihood, x0=guessParams, 
                            method='Nelder-Mead')
        self.optParams=optObj['x']
        #self.optParams=guessParams
        
        #aux for creating directories
        
        auxAcc=str(self.acc_model).split(' ')
        auxAcc=auxAcc[0]
        self.acc_model_name=auxAcc.split('.')
        self.acc_model_name=self.acc_model_name[2]
            
        auxRetr=str(self.retr_model).split(' ')
        auxRetr=auxRetr[0]
        self.retr_model_name=auxRetr.split('.')
        self.retr_model_name=self.retr_model_name[2]
        
        #Create directory to save outputs
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        #Create directory to save objects and h5 tracking file
        if not os.path.exists(self.directory+'obj/'):
            os.makedirs(self.directory+'obj/')
            
        self.modelName=self.acc_model_name+'_'+self.retr_model_name
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
        
    def priors(self,params,times):
        
        #errorbar has to be positive
        #errorbar: float = params["errorbar"]
        
        #if errorbar < 0: #prior on the variance (i.e. the error bars)
        #    return False
        
        
        # keep retreat rate below 20 mm/yr (why? bc we can't control lag thickness)
        if any(self.tr.retrModel.get_retreat_at_t(self.tr.retrModel._times) > (20*10**(-3))):
            return False
        
        #retreat rate should >=0 
        ret_t=self.tr.retrModel.get_retreat_at_t(self.tr.accuModel._times)
        if any(ret_t < 0):
            return False
        
        
        #depth of trough migration points should between 0 and -2 km
        if any(self.tr.ynear < -2e3) or any(self.tr.ynear > 0):
            return False
        
        #accumulation rate should >=0 
        acc_t=self.tr.accuModel.get_accumulation_at_t(self.tr.accuModel._times)
        if any(acc_t < 0):
            return False
        
        if any(self.tr.accuModel.get_accumulation_at_t(self.tr.accuModel._times) > (10*10**(-3))):
            return False
        
        #exponent of accumulation, if it exists, should be larger than -3
        if "acc_exponent" in params.keys():
            exponent: float = params["acc_exponent"]
            
            if exponent < -3:
                return False
        
        return True
        

    def ln_likelihood(self,params: Dict[str,float]):
        
        #set trough model with candidate parameters
        self.tr.set_model(params)
        #compute likelihood of model 
        likelihood_of_model=self.tr.lnlikelihood(self.xdata,self.ydata,
                                                 self.tr.accuModel._times)
        if self.priors(params,self.tr.accuModel._times):
            return likelihood_of_model
        else:
            return -1e99
    
    #And the negative of the log likelihood
    def neg_ln_likelihood(self,paramsArray):
        
        params=dict(zip(self.parameter_names, paramsArray))
        
        return -self.ln_likelihood(params)

    

class hardAgePriorMCMC(MCMC):
    
    def priors(self,params,times):
        otherPriors=super().priors(params,times)
        
        #age of trough should be less than 3 Myr
        if np.max(self.tr.timesxy)>3e6:
            return False
        
        return otherPriors*True
    
class softAgePriorMCMC(MCMC):
    def __init__(
        self,
        meanAge: float,
        maxSteps: int,
        thin_by: int,
        directory: str,
        tmp: int,
        acc_model = Union[str, Model],
        retr_model = Union[str, Model],
        errorbar = np.sqrt(1.6), #errorbar in pixels on the datapoints
        angle= 5.0
    ):
        self.meanAge=meanAge
        
        super().__init__(
            maxSteps,
            thin_by,
            directory,
            tmp,
            acc_model,
            retr_model,
            errorbar,
            angle)
        
    
    def priors(self,params,times):
        otherPriors=super().priors(params,times)
        if otherPriors==False:
            return False
        else:
            #age of trough is preferred to be 3 Myr 
            import scipy.stats as stat
            stdAge=5e5
            priorDistAge=stat.norm(self.meanAge,stdAge) #mean 3 Myr, std 5e5 y
            
            ageTrough=np.max(self.tr.timesxy)
            
            if ageTrough >= self.meanAge:
                priorAge=priorDistAge.sf(ageTrough)
            else:
                priorAge=priorDistAge.cdf(ageTrough)
                
            return otherPriors*priorAge
    
    def ln_likelihood(self,params: Dict[str,float]):
        
        #set trough model with candidate parameters
        self.tr.set_model(params)
        #compute likelihood of model 
        likelihood_of_model=self.tr.lnlikelihood(self.xdata,self.ydata,
                                                 self.tr.accuModel._times)
        
        priorValue=self.priors(params,self.tr.accuModel._times)
        
        if priorValue==False:
            return -1e99
        else:
            return np.log(priorValue)+likelihood_of_model
        
        