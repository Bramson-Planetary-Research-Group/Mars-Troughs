#!/usr/bin/env python                                                                                                                                  
import argparse
import pickle
import numpy as np
import mars_troughs as mt
from mars_troughs.custom_lag_models import ConstantLag,LinearLag,QuadraticLag,CubicLag,PowerLawLag
from mars_troughs.custom_acc_models import Linear_Insolation, Quadratic_Insolation,Cubic_Insolation,PowerLaw_Insolation
from mars_troughs.datapaths import load_insolation_data

p=argparse.ArgumentParser(description='Parse submodel numbers for MCMC')
p.add_argument("-acc_modelNumber",default=1,type=int,help="Number of the accumulation model")
p.add_argument("-lag_modelNumber",default=1,type=int,help="Number of the lag model")
p.add_argument("-maxSteps",default=100,type=int,help="Number of steps for MCMC")
args=p.parse_args()

acc_model_dict= { 1: Linear_Insolation,
                  2: Quadratic_Insolation,
                  3: Cubic_Insolation,
                  4: PowerLaw_Insolation }

lag_model_dict= { 1: ConstantLag,
                  2: LinearLag,
                  3: QuadraticLag,
                  4: CubicLag,
                  5: PowerLawLag }

tmp=2
(insolations,ins_times) = load_insolation_data(tmp)
ins_times=-ins_times
ins_times[0]=1e-10
acc_model=acc_model_dict[args.acc_modelNumber](ins_times,insolations)
lag_model=lag_model_dict[args.lag_modelNumber]()

maxSteps=args.maxSteps
subIter=maxSteps/10
directory='../../outputMCMC/insolation/TMP2/'
errorbar=np.sqrt(1.6)
angle=5.0
mcmcobj=mt.MCMC(maxSteps,subIter,directory,tmp,acc_model,lag_model, None, None, errorbar, angle)

filename=mcmcobj.filename

outfile=open(filename+'obj','wb')
pickle.dump(mcmcobj,outfile)
outfile.close()

print(filename)
