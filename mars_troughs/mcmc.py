#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 09:31:34 2021

@author: kris
"""
#import modules
import time
import numpy as np
import scipy.optimize as op
import emcee
import os
import sys
from mars_troughs import Trough
from typing import  Dict

class MCMC():
    """
    Class for running MCMC chains to obtain a sample of climate parameters
    of trough migration.
    """
    def __init__(
        self,
        xdata,
        ydata,
        troughObject: Trough,
        maxSteps: int,
        thin_by: int,
        directory: str,
    ):
        self.xdata=xdata
        self.ydata=ydata
        self.tr=troughObject
        self.maxSteps = maxSteps
        self.thin_by = thin_by
        self.directory = directory
        
        
        self.parameter_names = ([key for key in self.tr.all_parameters])
        
        #Find number of dimensions and number of parameters per submodel
        self.ndim=len(self.parameter_names)
        self.nwalkers=self.ndim*4
        
        
        #Linear optimization
        guessParams=np.array( list(self.tr.accuModel.parameters.values())
                             +list(self.tr.lagModel.parameters.values()))
        optObj= op.minimize(self.neg_ln_likelihood, x0=guessParams, 
                            method='Nelder-Mead')
        self.optParams=optObj['x']
        #self.optParams=guessParams
        
        #aux for creating directories
        auxAcc=str(self.tr.accuModel).split(' ')
        auxAcc=auxAcc[0]
        self.acc_model_name=auxAcc.split('.')
        self.acc_model_name=self.acc_model_name[2]
            
        auxLag=str(self.tr.lagModel).split(' ')
        auxLag=auxLag[0]
        self.lag_model_name=auxLag.split('.')
        self.lag_model_name=self.lag_model_name[2]
        
        #Create directory to save outputs
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        #Create directory to save objects and h5 tracking file
        if not os.path.exists(self.directory+'obj/'):
            os.makedirs(self.directory+'obj/')
        
        #define name of model and filename
        self.modelName=self.acc_model_name+'_'+self.lag_model_name
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
        
        #lag thickness has to be larger than 1e-15 mm and less than 20 mm
        if any(self.tr.lag_at_t  < 1e-15) or any(self.tr.lag_at_t > 20):
            return False
        #model tmp has to be between 0 and 2 km deep
        if np.min(self.tr.ynear) < -2e3 or np.max(self.tr.ynear)>0:
            return False
        
        #accumulation rate should >=0
        acc_t=self.tr.accuModel.get_accumulation_at_t(
                                                    self.tr.accuModel._times)
        if any(acc_t <= 0):
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
        likelihood_of_model=self.tr.lnlikelihood(
                                        self.xdata,
                                        self.ydata,
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
        xdata,
        ydata,
        troughObject,
        maxSteps: int,
        thin_by: int,
        directory: str,
    ):
        self.meanAge=meanAge
        
        super().__init__(
            xdata,
            ydata,
            troughObject,
            maxSteps,
            thin_by,
            directory)
        
    
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
        
        