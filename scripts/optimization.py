#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 17:21:47 2023

@author: kris


"""

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


class Optimization():
    
    def __init__(
          self,
          acc_model = Union[str, Model],
          lag_model = Union[str, Model],
          xdata,
          ydata,
          tmp,
          ):
    
        """
        Function for optimizing the climate parameters according to the obs tmp.
        """
        #load retreat data
        retreat_times, retreats, lags = load_retreat_data(tmp)
        retreat_times=-retreat_times
        ret_data_spline = RBS(lags, retreat_times, retreats)
        
        # Create  trough object 
        self.tr = mt.Trough(acc_model,lag_model,ret_data_spline,angle)
        
        parameter_names = ([key for key in self.tr.all_parameters])
        
        #Linear optimization
        guessParams=np.array( list(self.tr.accuModel.parameters.values())
                             +list(self.tr.lagModel.parameters.values()))
        optObj= op.minimize(neg_ln_likelihood, x0=guessParams, 
                            method='Nelder-Mead')
        optParams=optObj['x']
        
        #check that optimal parameters fit priors
        iparams=dict(zip(self.tr.all_parameter_names,optParams))
        likeInit=self.ln_likelihood(xdata,ydata,iparams)
        
        if likeInit==-1e99:
            print('Optimal parameters do not fit priors')
   
        #get model tmp from opt parameters
        xmodel=self.tr.xnear
        ymodel=self.tr.ynear
        
        #get mean accumulation
        meanAcc=np.mean(self.tr.accuModel.get_accumulation_at_t(
            newmcmc.tr.accuModel._times))
        
        #get age from opt parameters
        xmodelfull,ymodelfull=np.array(
            self.tr.get_trajectory(self.tr.accuModel._times))
        deepestIndex = np.where(ymodelfull == ymodel[-1])[0]
        age=self.tr.accuModel._times[deepestIndex]
        
        #get misfit
        xvar, yvar = (self.tr.errorbar * self.tr.meters_per_pixel) ** 2
        chi2 = (xdata - xmodel) ** 2 / xvar + (ydata - 
                                                      ymodel) ** 2 / yvar
        
        
        return optParams, xmodel, ymodel, meanAcc, age, chi2
    
    def neg_ln_likelihood(self,parameter_names,paramsArray):
        
        params=dict(zip(parameter_names, paramsArray))
        
        return -self.ln_likelihood(xdata,ydata,params)
        
    def ln_likelihood(self,xdata,ydata,params: Dict[str,float]):
        
        #set trough model with candidate parameters
        self.tr.set_model(params)
        #compute likelihood of model 
        likelihood_of_model=self.tr.lnlikelihood(xdata,ydata,
                                                 self.tr.accuModel._times)
        if self.priors(params,self.tr.accuModel._times):
            return likelihood_of_model
        else:
            return -1e99
        
    def priors(self,params,times):
        
        #lag thickness has to be larger than 1e-15 mm and less than 20 mm
        if any(self.tr.lag_at_t  < 1e-15) or any(self.tr.lag_at_t > 20):
            return False
        
        #depth of trough migration points should between 0 and -2 km
        if any(self.tr.ynear < -2e3) or any(self.tr.ynear > 0):
            return False
        
        #accumulation rate should >=0
        acc_t=self.tr.accuModel.get_accumulation_at_t(self.tr.accuModel._times)
        if any(acc_t <= 0):
            return False
        
        return True
        
    
    
    
    
    
    
    







