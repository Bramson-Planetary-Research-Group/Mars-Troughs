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
import os
import argparse

p=argparse.ArgumentParser(description="TMP data")
p.add_argument("-tmp",type=int,help="TMP 1 or 2")
args=p.parse_args()

listObj_ins1=glob.glob("../../outputMCMC/insolation/TMP1/obj/*obj")
listObj_ins1.sort()
listObj_ins2=glob.glob("../../outputMCMC/insolation/TMP2/obj/*obj")
listObj_ins2.sort()
listObj_obl1=glob.glob("../../outputMCMC/obliquity/TMP1/obj/*obj")
listObj_obl1.sort()
listObj_obl2=glob.glob("../../outputMCMC/obliquity/TMP2/obj/*obj")
listObj_obl2.sort()

if args.tmp==1:
    listObj=listObj_ins1+listObj_obl1
else:
    listObj=listObj_ins2+listObj_obl2

numfiles=len(listObj)

BICs=np.zeros(numfiles)
modelNames=['Models']

for i in np.arange(0,numfiles):

    infile=open(listObj[i],'rb')
    newmcmc=pickle.load(infile)
    infile.close()
    #get number of parameters
    numparams=len(newmcmc.parameter_names)
    
    #get number of tmp data points
    n=len(newmcmc.ydata)
    
    #compute BIC
    BIC=newmcmc.logprob.max()-1/2*numparams*np.log(n)
    BICs[i]=BIC
    modelNames=modelNames+[newmcmc.modelName]
    
    
maxBIC=np.argmax(BICs)
    
plt.figure
plt.plot(np.arange(1,numfiles+1),BICs)
plt.xticks(np.arange(1,numfiles+1),modelNames[1:],rotation = 90)
plt.xlabel('Model')
plt.ylabel('BIC')
plt.title(modelNames[maxBIC+1])
plt.subplots_adjust(bottom=0.6)


plt.savefig('../../outputMCMC/BICs/TMP' +str(args.tmp)+'.pdf',
            facecolor='w',pad_inches=0.1)

#save bic obj

outfile=open('../../outputMCMC/BICs/objTMP'+str(args.tmp),'wb')
pickle.dump(BICs,outfile)
pickle.dump(modelNames,outfile)
outfile.close()