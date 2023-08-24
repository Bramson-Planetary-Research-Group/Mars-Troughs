#!/usr/bin/env python                                                                                                                                  
import argparse
import pickle
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
                          load_obliquity_data,
                          load_retreat_data,
                          load_angle,
                          load_TMP_data)
import sys
from scipy.interpolate import RectBivariateSpline as RBS

def main():
        
    p=argparse.ArgumentParser(description='Parse submodel numbers for MCMC')
    p.add_argument("-acc",default=1,type=int,help="Number of the accumulation model")
    p.add_argument("-lag",default=1,type=int,help="Number of the lag model")
    p.add_argument("-steps",default=100,type=int,help="Number of steps for MCMC")
    p.add_argument("-thin_by",default=1,type=int,help="Skip iterations in ensemble")
    p.add_argument("-data", default="insolation",type=str, help="insolation or obliquity")
    p.add_argument("-trough", default= "1", type=str, help="trough number")
    p.add_argument("-tmp", default= 'all', type=str, help="tmp number or all")
    p.add_argument("-dir",default="../../outputMCMC/",type=str, help="directory for output")
    args=p.parse_args()
    
    #rename inputs
    tmp = args.tmp
    trough = args.trough
    maxSteps=args.steps
    directory= (args.dir + args.data + '/TMP' + str(tmp) + '/')
    thin_by=args.thin_by
    
    #Diccionaries for submodels
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
    
    #if data is insolation,load insolation data
    if args.data=="insolation":
        (insolations,times) = load_insolation_data(trough)
        times=-times.astype(float)
        times[0]=1e-10
        #create accumulation model
        acc_model=accModel_ins_dict[args.acc](times,insolations)
    else:
        #load obliquity
        (obliquity,times) = load_obliquity_data()
        times=-times.astype(float)
        times[0]=1e-10
        #create accumulation model
        acc_model=accModel_obl_dict[args.acc](times, obliquity)

    #create lag model
    lag_model=lagModel_dict[args.lag]()
    
    #load retreat data
    retreat_times, retreats, lags = load_retreat_data(trough)
    retreat_times=-retreat_times
    ret_data_spline = RBS(lags, retreat_times, retreats)

    #load tmp data  
    xdata,ydata=load_TMP_data(trough,tmp)
    
    #load trough angle
    angle=load_angle(trough)
    
    # Create  trough object 
    tr = mt.Trough(acc_model,lag_model,
                   ret_data_spline,angle)
    
    mcmcobj=mt.MCMC(xdata,ydata,
                    tr,
                    maxSteps,thin_by,directory)
    
    #save mcmc object
    filename=mcmcobj.filename
    outfile=open(filename,'wb')
    pickle.dump(mcmcobj,outfile)
    outfile.close()
    
    print(filename)
    
def mainArgs(acc,lag,steps,thin_by,data,trough,tmp,dir):
    sys.argv = ['mainInv.py', 
                '-acc', str(acc),
                '-lag', str(lag),
                '-steps',str(steps),
                '-thin_by', str(thin_by),
                '-data', str(data),
                '-trough', str(trough),
                '-tmp', str(tmp),
                '-dir', str(dir)]
    
    #cProfile.run('main()')
    main()

if __name__=='__main__':
    #cProfile.run('main()')
    main()
