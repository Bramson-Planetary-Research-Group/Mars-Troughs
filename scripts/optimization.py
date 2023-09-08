#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 17:21:47 2023

@author: kris


"""

import mars_troughs as mt
from mars_troughs import (Model, load_retreat_data)
import numpy as np
from scipy.interpolate import RectBivariateSpline as RBS
import scipy.optimize as op
from typing import Dict, Union


class Optimization():
    
    def __init__(
          self,
          tmp,
          angle,
          acc_model = Union[str, Model],
          lag_model = Union[str, Model],
          ):
    
        """
        Function for optimizing the climate parameters according to the obs tmp.
        """
        self.tmp=tmp
        #load retreat data
        retreat_times, retreats, lags = load_retreat_data(self.tmp)
        retreat_times=-retreat_times
        ret_data_spline = RBS(lags, retreat_times, retreats)
        
        # Create  trough object 
        self.tr = mt.Trough(acc_model,lag_model,ret_data_spline,angle)
        self.parameter_names = ([key for key in self.tr.all_parameters])
        
    def optFitSubTMP(self,xdata,ydata):
        
        
        self.xdata=xdata
        self.ydata=ydata
        
        #Use default params as initial params in opt
        self.prevOptParams=[0.0015,-0.75,90,0.5,0.20] #results from paper for all 16 points tmp1
        initparams=dict(zip(self.parameter_names, self.prevOptParams))
        self.tr.set_model(initparams)
        
        #get init model tmp
        xinit, yinit, timesxy = self.tr.get_nearest_points(
                                                      self.xdata, self.ydata,
                                                      self.tr.accuModel._times)
        
        #optimize parameters from fit to subsample of TMP
        optObj= op.minimize(self.neg_ln_likelihood, x0=self.prevOptParams, 
                            method='Nelder-Mead')
        optParams=optObj['x']
        
        #check that optimal parameters fit priors
        iparams=dict(zip(self.tr.all_parameter_names,optParams))
        likeInit=self.ln_likelihood(iparams)
        
        if likeInit==-1e99:
            print('Optimal parameters do not fit priors')
   
        #get model tmp from opt parameters
        xmodel=self.tr.xnear
        ymodel=self.tr.ynear
        
        #get age from opt parameters
        xmodelfull,ymodelfull=np.array(
            self.tr.get_trajectory(self.tr.accuModel._times))
        deepestIndex = np.where(ymodelfull == ymodel[-1])[0][0]
        age=self.tr.accuModel._times[deepestIndex]
        
        #get mean accumulation from opt parameters
        acc=self.tr.accuModel.get_accumulation_at_t(
            self.tr.accuModel._times)
        meanAcc=np.mean(acc[:deepestIndex])*1000 #mm/yr
        
        #get misfit from opt params
        xvar, yvar = (self.tr.errorbar * self.tr.meters_per_pixel) ** 2
        chi2 = (self.xdata - xmodel) ** 2 / xvar + (self.ydata - 
                                                      ymodel) ** 2 / yvar
        
        chi2sum=chi2.sum()
        return xinit, yinit, optParams, xmodel, ymodel, meanAcc, age, chi2sum
    
    def neg_ln_likelihood(self,paramsArray):
        
        params=dict(zip(self.parameter_names, paramsArray))
        
        return -self.ln_likelihood(params)
        
    def ln_likelihood(self,params: Dict[str,float]):
        
        #set trough model with candidate parameters
        self.tr.set_model(params)
        #compute likelihood of model 
        likelihood_of_model=self.tr.lnlikelihood(self.xdata,self.ydata,
                                                 self.tr.accuModel._times)
        if self.priors():
            return likelihood_of_model
        else:
            return -1e99
        
    def priors(self):
        
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
        
    
    
    
    
    
    
    







