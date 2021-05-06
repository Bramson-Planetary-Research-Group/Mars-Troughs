#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:37:51 2021

@author: laferrierek

Linear model for both accumulation and lag
"""
from mars_troughs.model import Model
from typing import Dict

class LinearModel(Model):
    intercept: float=1.0
    slope: float=1.0
    
    @property
    def parameter_names(self) -> Dict[str, float]:
        return["intercept", "slope"]
    
    def eval(self, x):
        return self.intercept + x*self.slope
    

class QuadModel(Model):
    intercept: float = 1.0,
    linearCoeff: float = 1e-6,
    quadCoeff: float = 1e-6,

    @property
    def parameters_names(self) -> Dict[str, float]:
        return['intercept', 'linearCoeff', 'quadCoeff']
                 
    def eval(self, x):
        p = [self.intercept, self.linearCoeff, self.quadCoeff]
        return sum((a*x**i for i,a in enumerate(p)))
    
