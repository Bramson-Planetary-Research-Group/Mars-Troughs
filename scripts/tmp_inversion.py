#!/usr/bin/env python                                                                                                                                  
import argparse
import pickle
import numpy as np
import mars_troughs as mt
from mars_troughs import (ConstantLag,
                          LinearLag,
                          QuadraticLag,
                          CubicLag,
                          PowerLawLag)
from mars_troughs import (Linear_Insolation, 
                          Quadratic_Insolation,
                          Cubic_Insolation,
                          PowerLaw_Insolation)
from mars_troughs import (Linear_Obliquity, 
                          Quadratic_Obliquity,
                          Cubic_Obliquity,
                          PowerLaw_Obliquity)
from mars_troughs import (load_insolation_data, 
                          load_obliquity_data)
from mars_troughs import DATAPATHS
import sys

def main():
        
    p=argparse.ArgumentParser(description='Parse submodel numbers for MCMC')
    p.add_argument("-acc",default=1,type=int,help="Number of the accumulation model")
    p.add_argument("-lag",default=1,type=int,help="Number of the lag model")
    p.add_argument("-steps",default=100,type=int,help="Number of steps for MCMC")
    p.add_argument("-thin_by",default=1,type=int,help="Skip iterations in ensemble")
    p.add_argument("-data", default="insolation",type=str, help="insolation or obliquity")
    p.add_argument("-tmp", default=1,type=int, help="tmp number")
    p.add_argument("-dir",default="../../outputMCMC/",type=str, help="directory for output")
    args=p.parse_args()
    
    accModel_ins_dict= { 1: Linear_Insolation,
                         2: Quadratic_Insolation,
                         3: Cubic_Insolation,
                         4: PowerLaw_Insolation }
    
    accModel_obl_dict= { 1: Linear_Obliquity,
                         2: Quadratic_Obliquity,
                         3: Cubic_Obliquity,
                         4: PowerLaw_Obliquity }
    
    lagModel_dict= {  1: ConstantLag,
                      2: LinearLag,
                      3: QuadraticLag,
                      4: CubicLag,
                      5: PowerLawLag }
    
    tmp=args.tmp
    
    #if data is insolation,load insolation data
    if args.data=="insolation":
        (insolations,times) = load_insolation_data(tmp)
        times=-times.astype(float)
        times[0]=1e-10
        acc_model=accModel_ins_dict[args.acc](times,insolations)
        
        
    else:
        (obliquity,times) = load_obliquity_data()
        times=-times.astype(float)
        times[0]=1e-10
        acc_model=accModel_obl_dict[args.acc](times, obliquity)

    
    #create lag model
    lag_model=lagModel_dict[args.lag]()
    
    maxSteps=args.steps
    directory= (args.dir + args.data + '/TMP' + str(tmp) + '/')
    
    if tmp==1:
        angle=2.9
        #get tmp data
        xdata,ydata=np.loadtxt(DATAPATHS.TMP1, 
                                         unpack=True) #Observed TMP data   
        xdata=xdata*1000 #km to m 
        
        
    elif tmp==2:
        angle=1.9
        #get tmp data
        xdata,ydata=np.loadtxt(DATAPATHS.TMP2, 
                                     unpack=True) #Observed TMP data  
        xdata=xdata*1000 #km to m 
    
    thin_by=args.thin_by
    mcmcobj=mt.MCMC(maxSteps,thin_by,directory,tmp,xdata,ydata,acc_model,
                    lag_model,
                    angle)
    
    filename=mcmcobj.filename
    
    outfile=open(filename,'wb')
    pickle.dump(mcmcobj,outfile)
    outfile.close()
    
    print(filename)
    
def mainArgs(acc,lag,steps,thin_by,data,tmp,dir):
    sys.argv = ['mainInv.py', 
                '-acc', str(acc),
                '-lag', str(lag),
                '-steps',str(steps),
                '-thin_by', str(thin_by),
                '-data', str(data),
                '-tmp', str(tmp),
                '-dir', str(dir)]
    
    #cProfile.run('main()')
    main()

if __name__=='__main__':
    #cProfile.run('main()')
    main()
