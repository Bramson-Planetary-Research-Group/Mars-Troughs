#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 10:32:35 2021

@author: kris
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
import glob
import argparse
import sys
from numpy import unravel_index

def main():

    p=argparse.ArgumentParser(description="TMP data")
    p.add_argument("-tmp",type=int,help="TMP 1 or 2")
    p.add_argument("-initmodel",type=int,help="initial model for plotting")
    p.add_argument("-stepEnsemble",type=int,help="skip models for plotting")
    args=p.parse_args()
    
    listObj_ins1=glob.glob("../../outputMCMC/insolation/TMP1/obj/*obj*")
    listObj_ins1.sort()
    listObj_ins2=glob.glob("../../outputMCMC/insolation/TMP2/obj/*obj*")
    listObj_ins2.sort()
    listObj_obl1=glob.glob("../../outputMCMC/obliquity/TMP1/obj/*obj*")
    listObj_obl1.sort()
    listObj_obl2=glob.glob("../../outputMCMC/obliquity/TMP2/obj/*obj*")
    listObj_obl2.sort()
    
    if args.tmp==1:
        listObj=listObj_ins1+listObj_obl1
    else:
        listObj=listObj_ins2+listObj_obl2
    
    numfiles=len(listObj)
    
    BICs=np.zeros(numfiles)
    ages=np.zeros(numfiles)
    modelNames=['Models']
    
    print(numfiles)
    
    for i in np.arange(0,numfiles):
    
        infile=open(listObj[i],'rb')
        newmcmc=pickle.load(infile)
        infile.close()
        #get number of parameters
        numparams=len(newmcmc.parameter_names)
        #get number of tmp data points
        n=len(newmcmc.ydata)
        
        #get params of highest likelihood 
        highlike=newmcmc.logprob.argmax()
        highlike2d=unravel_index(highlike,newmcmc.logprob.shape)
        iparams=newmcmc.samples[highlike2d[0],highlike2d[1],:]
        iparamsdic=dict(zip(newmcmc.tr.all_parameter_names,iparams))
        
        #get tmp and age
        newmcmc.tr.set_model(iparamsdic)
        tmp=np.array(newmcmc.tr.get_trajectory(newmcmc.tr.accuModel._times))

        #find ages
        lastxdata=newmcmc.xdata[n-1]
        lastydata=newmcmc.ydata[n-1]
    
        xi=tmp[0,:]
        yi=tmp[1,:]
        disti = newmcmc.tr._L2_distance(xi, lastxdata, yi, lastydata)
        ind = np.argmin(disti)
        age = newmcmc.tr.accuModel._times[ind]/1000000
        
        #compute BIC
        BIC=newmcmc.logprob.max()-1/2*numparams*np.log(n)
        #save BIC and age
        BICs[i]=BIC
        ages[i]=age
        modelNames=modelNames+[newmcmc.modelName]
        print(i)
        
        
    maxBIC=np.argmax(BICs)
    bestAge=ages[maxBIC]
    
    plt.close('all')
    plt.figure
    plt.plot(ages,BICs,marker='.',linestyle="None")
    plt.xlabel('Age (Myr')
    plt.ylabel('BIC')
    plt.title(modelNames[maxBIC]+'_'+str(bestAge)+' Myr')
    plt.subplots_adjust(bottom=0.6)
    
    
    plt.savefig('../../outputMCMC/BICs/TMP' +str(args.tmp)+'age.pdf',
                facecolor='w',pad_inches=0.1)
    
    #save bic obj
    
    outfile=open('../../outputMCMC/BICs/objTMP'+str(args.tmp)+'age','wb')
    pickle.dump(BICs,outfile)
    pickle.dump(modelNames,outfile)
    outfile.close()
    
def mainArgs(tmp,initmodel,stepEnsemble):
     sys.argv = ['bic.py',
                 '-tmp', str(tmp),
                 '-initmodel',str(initmodel),
                 '-stepEnsemble',str(stepEnsemble)]
     main()
     
if __name__=="__main__":
    main()
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     