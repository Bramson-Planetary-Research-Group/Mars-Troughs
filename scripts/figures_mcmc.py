#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 20:37:17 2021

@author: kris
"""
import numpy as np
import corner
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import glob
import sys
from mars_troughs.datapaths import load_insolation_data, load_obliquity_data
import math

def main():
    p=argparse.ArgumentParser(description="filename for plotting")
    p.add_argument("-objpath",type=str,help="name file or dir \
                                           for loading mcmc object")
    p.add_argument("-plotdir",type=str,help="name dir for saving plots")
    p.add_argument("-nmodels",type=int,help="nmodels for ensemble")
    p.add_argument("-stepEnsemble",type=int,help="skip models for \
                                                  plotting")
    args=p.parse_args()
    
    #Check if path is file or directory
    
    if '_Obliquity_' in args.objpath:
        listObj=[args.objpath]
        numfiles=len(listObj)
        print('Input file path')
    elif '_Insolation_' in args.objpath:
        listObj=[args.objpath]
        numfiles=len(listObj)
        print('Input file path')
    else:
        listObj=glob.glob(args.objpath+'*obj*')
        numfiles=len(listObj)
        print('Input directory path')
        print(numfiles,' files')  
        
    #loop figures script over numfiles (1 if path is file) 
        
    for ifile in np.arange(0,numfiles):
            
        infile=open(listObj[ifile],'rb')
        print(listObj[ifile])
        newmcmc=pickle.load(infile)
        infile.close()

        #create folder for saving figures
        if not os.path.exists(args.plotdir+'figures/'):
            os.makedirs(args.plotdir+'figures/')
        
        #set parameters for plotting
        paramsList=list(newmcmc.tr.all_parameter_names)
        numparams=len(paramsList)
        
        #subsample ensemble
        ensemble=newmcmc.samples[-1*args.nmodels::args.stepEnsemble,:,:]
        xaxis=np.arange(newmcmc.totalSteps-(args.nmodels-1)*newmcmc.thin_by,newmcmc.totalSteps+1,args.stepEnsemble*newmcmc.thin_by)
        nmodels=len(xaxis) 
        logprob=newmcmc.logprob[-1*args.nmodels::args.stepEnsemble,:]

        #find model with highest likelihood
        indxbest2d=np.unravel_index(logprob.argmax(),logprob.shape)
        
        #parameter values per iteration---------------------------------
        plt.figure()
        
        for i in np.arange(1,numparams):
            plt.subplot(numparams,1,i)
            plt.plot(xaxis,ensemble[:,:,i-1])
            plt.plot(xaxis[indxbest2d[0]],ensemble[indxbest2d[0],
                                     indxbest2d[1],i-1],marker='*')
            plt.xticks([], [])
            plt.title(paramsList[i-1])
            
        plt.subplot(numparams,1,numparams)
        plt.plot(xaxis,ensemble[:,:,numparams-1])
        plt.plot(xaxis[indxbest2d[0]],ensemble[indxbest2d[0],
                                     indxbest2d[1],numparams-1],marker='*')
        plt.title(paramsList[numparams-1])
        plt.xlabel('Step')
        
        #create folder for saving figure
        if not os.path.exists(args.plotdir+'figures/'+'paramsIter/'):
            os.makedirs(args.plotdir+'figures/'+'paramsIter/')
            
        plt.savefig(args.plotdir+'figures/'+'paramsIter/'
                    +newmcmc.modelName+'_'+str(newmcmc.maxSteps)+'.pdf',
                    facecolor='w',pad_inches=0.1)
        
        #corner plot posterior ----------------------------------------------
        #reshape ensemble
        ensemble2d=ensemble.reshape((newmcmc.nwalkers*nmodels,numparams))
            
        #plot
        corner.corner(ensemble2d,labels=paramsList,quiet=True)
        
        #create folder for saving figure
        if not os.path.exists(args.plotdir+'figures/'+'corner/'):
            os.makedirs(args.plotdir+'figures/'+'corner/')
            
        plt.savefig(args.plotdir+'figures/'+'corner/'
                    +newmcmc.modelName+'_'+str(newmcmc.maxSteps)+'.pdf',
                    facecolor='w',pad_inches=0.1)
        

        #log likelihood -------------------------------------------------------
        
        #get likelihood of opt params, init params and best fit params
        #opt params
        optdict=dict(zip(newmcmc.tr.all_parameter_names,newmcmc.optParams))
        optlike=newmcmc.ln_likelihood(optdict)
        #init params
        initlike=np.zeros((newmcmc.nwalkers,1))
        for i in range(0,newmcmc.nwalkers):
            initdict=dict(zip(newmcmc.tr.all_parameter_names,
                              newmcmc.initParams[i,:]))
            initlike[i]=newmcmc.ln_likelihood(initdict)
        #best fit params
        bestparams=ensemble[indxbest2d[0],indxbest2d[1],:]
        bestdict=dict(zip(newmcmc.tr.all_parameter_names,bestparams))
        bestlike=newmcmc.ln_likelihood(bestdict)
        
        #plot likelihood from step 1 to final step
        xaxisLike=np.arange(0,newmcmc.totalSteps,newmcmc.thin_by)
        
        plt.figure()
        plt.plot(xaxisLike,newmcmc.logprob)
        plt.plot(0,initlike.T,marker='o',color='k')
        plt.plot(0,initlike[0],label='Init params',marker='o',color='k')
        plt.plot(-1,optlike,label='Opt params',marker='*',color='r')
        plt.plot(xaxis[indxbest2d[0]],bestlike,label='Best fit params',
                 marker='^',color='k')
        plt.title(label='mean acceptance ratio = '+ 
                  str(np.round(np.mean(newmcmc.accFraction),2)))
        plt.xlabel('Step')
        plt.ylabel('log prob')
        plt.legend()

        #create folder for saving figure
        if not os.path.exists(args.plotdir+'figures/'+'logprob/'):
            os.makedirs(args.plotdir+'figures/'+'logprob/')
            
        plt.savefig(args.plotdir+'figures/'+'logprob/'
                    +newmcmc.modelName+'_'+str(newmcmc.maxSteps)+'.pdf',
                    facecolor='w',pad_inches=0.1)
    
        
        #autocorrelation values-----------------------------------------------
        plt.figure()
        autoxaxis=(newmcmc.maxSteps/10)*np.arange(1,11)
        autoxaxis=autoxaxis[:len(newmcmc.autocorr)]
        
        plt.plot(autoxaxis,autoxaxis,"--k",label=r'Length chain')
        plt.plot(autoxaxis[np.nonzero(newmcmc.autocorr)],newmcmc.autocorr[np.nonzero(newmcmc.autocorr)],label=r'$\tau$ estimate')
        plt.xlabel('Iteration')
        ax=plt.gca()
        ax.legend()
        plt.yscale("log")        
        
        #create folder for saving figure
        if not os.path.exists(args.plotdir+'figures/'+'autocorr/'):
            os.makedirs(args.plotdir+'figures/'+'autocorr/')
            
        plt.savefig(args.plotdir+'figures/'+'autocorr/'
                    +newmcmc.modelName+'_'+str(newmcmc.maxSteps)+'.pdf',
                    facecolor='w',pad_inches=0.1)
        
        #lag, acc rate and y per time for each model ----------------------------
        #indxlagparams=paramsList.index(lagparamsList[0])
        acct=np.zeros((nmodels*newmcmc.nwalkers,len(newmcmc.tr.accuModel._times)))
        tmpt=np.zeros((nmodels*newmcmc.nwalkers,len(newmcmc.tr.accuModel._times),2))
        retreatt=np.zeros((nmodels*newmcmc.nwalkers,len(newmcmc.tr.accuModel._times)))

        indxw=0
        
        if "Lag" in str(newmcmc.tr.retrModel):
            lagt=np.zeros((nmodels*newmcmc.nwalkers,len(newmcmc.tr.accuModel._times)))
            
            for i in range(0,nmodels):
                for w in range(0,newmcmc.nwalkers):
                    iparams=dict(zip(newmcmc.tr.all_parameter_names,ensemble[i,w,:]))
                    newmcmc.tr.set_model(iparams)
                    
                    retreati=newmcmc.tr.retreat_model_t
                    lagti=newmcmc.tr.retrModel.get_lag_at_t(newmcmc.tr.accuModel._times)
                    accti=newmcmc.tr.accuModel.get_accumulation_at_t(newmcmc.tr.accuModel._times)
                    tmpti=np.array(newmcmc.tr.get_trajectory(newmcmc.tr.accuModel._times))
                    
                    retreatt[indxw]=retreati
                    lagt[indxw]=lagti
                    acct[indxw]=accti
                    tmpt[indxw,:,:]=tmpti.T
                    indxw=indxw+1

        elif "Retreat" in str(newmcmc.tr.retrModel):
        
            for i in range(0,nmodels):
                for w in range(0,newmcmc.nwalkers):
                    iparams=dict(zip(newmcmc.tr.all_parameter_names,ensemble[i,w,:]))
                    newmcmc.tr.set_model(iparams)
                    
                    retreati=newmcmc.tr.retrModel.get_retreat_at_t(newmcmc.tr.accuModel._times)
                    accti=newmcmc.tr.accuModel.get_accumulation_at_t(newmcmc.tr.accuModel._times)
                    tmpti=np.array(newmcmc.tr.get_trajectory(newmcmc.tr.accuModel._times))
                    
                    retreatt[indxw]=retreati
                    acct[indxw]=accti
                    tmpt[indxw,:,:]=tmpti.T
                    
                    indxw=indxw+1
      
        #get errorbar 1d
        errorbar1d=ensemble[:,:,0].reshape(nmodels*newmcmc.nwalkers,1)       
        #reshape log prob        
        logprob1d=logprob.reshape(nmodels*newmcmc.nwalkers,1)
        #best model indx
        indxbest=np.argmax(logprob1d)
                
        subsample=10
        timeaxis=newmcmc.tr.accuModel._times
        timesub=timeaxis[0::subsample]
        
        
        #plot retreat rates
        fig = plt.figure(figsize=(6, 6), dpi=300)

        ax1 = plt.subplot(3,1,1)
        plt.plot(timesub/1000000,retreatt[:,0::subsample].T*1000,c="gray",
                                            alpha=0.05, zorder=-1)
        plt.plot(timesub/1000000,retreatt[indxbest,0::subsample]*1000,c='#67a9cf')
        plt.hlines(0, 0, 5, colors='black', alpha=0.5, linestyle='dashed')

        ax1.set_xticklabels([])
        ax1.set_ylabel('R(t) (mm/year)')
       
        ymax = math.ceil(np.max(1000*retreatt[:, 0::subsample]))
        step = ymax/5
        ax1.set_yticks(np.arange(0, ymax+2*step, step))
        ax1.set_yticks(np.arange(0, ymax+2*step, step/5), minor=True)
        ax1.set_ylim((-1, ymax+step)) 
        
        ax1.set_xticks(np.arange(0, 5, 0.25), minor=True)
        ax1.set_xlim((0, 5))
        
        #plot acct
        ax2 = plt.subplot(3,1,2)
        plt.plot(timesub/1000000,1000*acct[:,0::subsample].T,c="gray",
                                            alpha=0.05, zorder=-1)
        plt.plot(timesub/1000000,1000*acct[indxbest,0::subsample],c="#d7301f")
        
        plt.hlines(0, 0, 5, colors='black', alpha=0.5, linestyle='dashed')

        ax2.set_ylabel('A(t) (mm/year)')
        ax2.set_xticklabels([])
        
        ymax = math.ceil(np.max(1000*acct[:, 0::subsample]))
        step = ymax/5
        ax2.set_yticks(np.arange(0, ymax+2*step, step))
        ax2.set_yticks(np.arange(0, ymax+2*step, step/5), minor=True)

        ax2.set_ylim((-1, ymax+step)) 
         
        ax2.set_xticks(np.arange(0, 5, 0.25), minor=True)

        ax2.set_xlim((0, 5))
        

        if '_Obliquity_' in newmcmc.modelName:
            #plot obliquity data
            data,times =  load_obliquity_data()
            titledata = r'Obliquity ($\degree$)'
        else:
            #plot insolation data
            data,times =  load_insolation_data(newmcmc.tmp)
            titledata = 'Insolation (W/$m^2$)'
        
        ax3 = plt.subplot(3,1,3)
        plt.plot(-1*times/1000000,data)
        plt.title(titledata)
        plt.xlabel('Time (Myr)')
        
        ymax = math.ceil(np.max(data))
        step = round(ymax/5)
        ax3.set_yticks(np.arange(0, ymax+2*step, step))
        ax3.set_yticks(np.arange(0, ymax+2*step, step/5), minor=True)
        ax3.set_ylim((0, ymax+step))
        
        ax3.set_xticks(np.arange(0, 5, 0.25), minor=True)
        
        ax3.set_xlim((0, 5))
        
        ax1.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                             bottom=True, top=True, left=True, right=True, direction='in')
        ax2.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                             bottom=True, top=True, left=True, right=True, direction='in')
        ax3.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                             bottom=True, top=True, left=True, right=True, direction='in')
        
        fig.tight_layout()
        
        #create folder for saving figure
        if not os.path.exists(args.plotdir+'figures/'+'ar_rates/'):
            os.makedirs(args.plotdir+'figures/'+'ar_rates/')
            
        plt.savefig(args.plotdir+'figures/'+'ar_rates/'
                    +newmcmc.modelName+'_'+str(newmcmc.maxSteps)+'.png',
                    facecolor='w',pad_inches=0.1)
        
        # tmp fit ---------------------------------------------------
        plt.figure(figsize=(6,6), dpi=250)
        bestTMP=tmpt[indxbest,:,:]
        plt.plot(bestTMP[:,0],bestTMP[:,1],c='#02818a')
        
        #get errorbar of best tmp
        bestErrorbar=errorbar1d[indxbest]
        
        ratioyx=0.4;
        
        #find nearest points
        x_model=bestTMP[:,0]
        y_model=bestTMP[:,1]
        xnear = np.zeros_like(newmcmc.xdata)
        ynear = np.zeros_like(newmcmc.ydata)
        timenear = np.zeros_like(newmcmc.xdata)
        
        
        for i, (xdi, ydi) in enumerate(zip(newmcmc.xdata, newmcmc.ydata)):
            dist = newmcmc.tr._L2_distance(x_model, xdi, y_model, ydi)
            ind = np.argmin(dist)
            xnear[i] = x_model[ind]
            ynear[i] = y_model[ind]
            timenear[i] = newmcmc.tr.accuModel._times[ind]
            
        #plot tmp data and errorbar
        xerr, yerr = bestErrorbar*newmcmc.tr.meters_per_pixel
        xerr = np.abs(xerr)
        yerr = np.abs(yerr)
        
        for i in range(nmodels*newmcmc.nwalkers):
            indx=i
            plt.plot(tmpt[indx,:,0],tmpt[indx,:,1],c="#bdc9e1", alpha=0.1, zorder=-1)
        plt.plot(tmpt[indx,:,0],tmpt[indx,:,1],c="#bdc9e1", alpha=0.1, zorder=-1,label='Ensemble models')

        #plot observed data with errorbars
        #sharad images error: 20 m per pixel vertically and 475 m per pixel
        #horizontally
        plt.errorbar(x=xnear, xerr=xerr, y=ynear, yerr=yerr, 
         c='#02818a', marker='.', ls='', label='Best model')
        
        plt.errorbar(x=newmcmc.xdata, 
                     xerr=newmcmc.tr.errorbar*newmcmc.tr.meters_per_pixel[0],
                     y=newmcmc.ydata, 
                     yerr=newmcmc.tr.errorbar*newmcmc.tr.meters_per_pixel[1], 
                     c='#d7301f', marker='.', ls='',label='Observed TMP')
        
        
        plt.xlabel("Horizontal distance (m)")
        plt.ylabel("Depth (m)")
        
        ax=plt.gca()
        ax.legend(bbox_to_anchor=(0.5, -0.3), loc='upper left')
        xmax=np.max(newmcmc.xdata)+1000
        ymin=np.min(newmcmc.ydata)-100
        ax.set_ylim(ymin,0)
        ax.set_xlim(0,xmax)
        ax.set_box_aspect(ratioyx)
        ax.minorticks_on()
        ax.yaxis.set_tick_params(which='minor', bottom=False)
        
        ax.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                             bottom=True, top=True, left=True, right=True, direction='in')
        
        
        #plot times on upper axis
        ax2=ax.twiny()
        color='#67a9cf'
        ax2.set_xlabel('Time before present ( Myr)',color=color)
        #plt.scatter(xnear,ynear,marker="o",color='b')
        
        ax2.set_ylim(ymin,0)
        ax2.set_xlim(0,xmax)
        ax2.tick_params(axis='x',labelcolor=color)
        plt.xticks(xnear,np.round(timenear/1000000,2).astype(float),rotation=90)
        ax2.set_box_aspect(ratioyx)
        
        #create folder for saving figure
        if not os.path.exists(args.plotdir+'figures/'+'tmp/'):
            os.makedirs(args.plotdir+'figures/'+'tmp/')
    
            
        plt.savefig(args.plotdir+'figures/'+'tmp/'
                    +newmcmc.modelName+'_'+str(newmcmc.maxSteps)+'.pdf',
                    facecolor='w',pad_inches=0.1)
        
        #save tmp full range of model---------------------------------
        plt.figure()
        for i in range(nmodels*newmcmc.nwalkers):
            indx=i
            plt.plot(tmpt[indx,:,0],tmpt[indx,:,1],c="gray", 
                     alpha=0.1, zorder=-1)
        plt.xlabel("Horizontal dist [m]")
        plt.ylabel("V. dist [m]")
        plt.errorbar(x=newmcmc.xdata, 
                     xerr=newmcmc.tr.errorbar*newmcmc.tr.meters_per_pixel[0], 
                     y=newmcmc.ydata, 
                     yerr=newmcmc.tr.errorbar*newmcmc.tr.meters_per_pixel[1], 
                     c='r', marker='.', ls='')
        
         #create folder for saving figure
        if not os.path.exists(args.plotdir+'figures/'+'tmpFull/'):
            os.makedirs(args.plotdir+'figures/'+'tmpFull/')
    
            
        plt.savefig(args.plotdir+'figures/'+'tmpFull/'
                    +newmcmc.modelName+'_'+str(newmcmc.maxSteps)+'.png',
                    facecolor='w',pad_inches=0.1)
        


        #plot hists of age-------------------------
        
        ndata=len(newmcmc.xdata)
        lastxdata=newmcmc.xdata[ndata-1]
        lastydata=newmcmc.ydata[ndata-1]
        ages = np.zeros((nmodels*newmcmc.nwalkers,1))

        for w in range(0,nmodels*newmcmc.nwalkers):
            xi=tmpt[w,:,0]
            yi=tmpt[w,:,1]
            disti = newmcmc.tr._L2_distance(xi, lastxdata, yi, lastydata)
            ind = np.argmin(disti)
            ages[w] = newmcmc.tr.accuModel._times[ind]/1000000
            
        
        plt.figure(figsize=(5,5), dpi=250)
        plt.hist(ages,bins=100, color="#02818a")
        plt.axvline(x=ages[indxbest],color='k',label='Age best model',
                    linestyle='dashed')

        counts, bins = np.histogram(ages, bins=100)
        plt.yticks(np.arange(0, round(np.max(counts), -1)+1, 10))
        plt.yticks(np.arange(0, round(np.max(counts), -1), 2), minor=True)
        plt.ylim((0, round(np.max(counts), -1)))
        plt.xticks(np.linspace(0, math.ceil(np.max(ages)), 5))
        plt.xticks(np.linspace(0, math.ceil(np.max(ages)), 10), minor=True)
        plt.gca().set_xlim(left=0, right=None)
        plt.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                             bottom=True, top=True, left=True, right=True, direction='in')
        plt.xlabel('Age (Myr)')
        plt.ylabel('# models')
        plt.legend()
        
        #create folder for saving figure
        if not os.path.exists(args.plotdir+'figures/'+'age/'):
            os.makedirs(args.plotdir+'figures/'+'age/')
    
        plt.savefig(args.plotdir+'figures/'+'age/'
                    +newmcmc.modelName+'_'+str(newmcmc.maxSteps)+'.pdf',
                    facecolor='w',pad_inches=0.1)
        
        
        # retreat and accumulation with age restricted solutions --------------
        age_x = max(ages)
        fig = plt.figure(figsize=(6, 6), dpi=300)
        ax1 = plt.subplot(3,1,1)
        plt.plot(timesub/1000000, 1000*retreatt[:, 0::subsample].T, c='gray', alpha=0.05, zorder=-1)
        plt.plot(timesub/1000000, 1000*retreatt[indxbest, 0::subsample], c='#67a9cf')
        plt.hlines(0, 0, 5, colors='black', alpha=0.5, linestyle='dashed')
        ax1.set_xticklabels([])
        ax1.set_ylabel('R(t) (mm/year)')
       
        ymax = math.ceil(np.max(1000*retreatt[:, 0::subsample]))
        step = ymax/5
        ax1.set_yticks(np.arange(0, ymax+2*step, step))
        ax1.set_yticks(np.arange(0, ymax+2*step, step/5), minor=True)
        ax1.set_ylim((-1, ymax+step)) 
        
        # restrict to age limit
        ax1.set_xticks(np.arange(0, 5, 0.25), minor=True)
        ax1.set_xlim((0, age_x))
        
        #plot acct
        ax2 = plt.subplot(3,1,2)
        plt.plot(timesub/1000000,1000*acct[:,0::subsample].T,c="gray",
                                            alpha=0.05, zorder=-1)
        plt.plot(timesub/1000000,1000*acct[indxbest,0::subsample],c="#d7301f")
        plt.hlines(0, 0, 5, colors='black', alpha=0.5, linestyle='dashed')

        ax2.set_ylabel('A(t) (mm/year)')
        ax2.set_xticklabels([])
        
        ymax = math.ceil(np.max(1000*acct[:, 0::subsample]))
        step = ymax/5
        ax2.set_yticks(np.arange(0, ymax+2*step, step))
        ax2.set_yticks(np.arange(0, ymax+2*step, step/4), minor=True)

        ax2.set_ylim((-1, ymax+step)) 
         
        ax2.set_xticks(np.arange(0, 5, 0.25), minor=True)

        ax2.set_xlim((0, age_x))
        
        if '_Obliquity_' in newmcmc.modelName:
            #plot obliquity data
            data,times =  load_obliquity_data()
            titledata = r'Obliquity ($\degree$)'
        else:
            #plot insolation data
            data,times =  load_insolation_data(newmcmc.tmp)
            titledata = 'Insolation (W/m^2)'
        
        ax3 = plt.subplot(3,1,3)
        plt.plot(-1*times/1000000,data, c='k')
        ax3.set_ylabel(titledata)
        plt.xlabel('Time (Myr)')

        ymax = math.ceil(np.max(data))
        step = round(ymax/5)
        ax3.set_yticks(np.arange(0, ymax+2*step, step))
        ax3.set_yticks(np.arange(0, ymax+2*step, step/5), minor=True)
        ax3.set_ylim((0, ymax+step))
        
        ax3.set_xticks(np.arange(0, 5, 0.25), minor=True)
        
        ax3.set_xlim((0, age_x))
        
        ax1.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                             bottom=True, top=True, left=True, right=True, direction='in')
        ax2.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                             bottom=True, top=True, left=True, right=True, direction='in')
        ax3.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                             bottom=True, top=True, left=True, right=True, direction='in')      
        
        fig.tight_layout()
        
        #create folder for saving figure
        if not os.path.exists(args.plotdir+'figures/'+'ar_rates_crop/'):
            os.makedirs(args.plotdir+'figures/'+'ar_rates_crop/')
            
        plt.savefig(args.plotdir+'figures/'+'ar_rates_crop/'
                    +newmcmc.modelName+'_'+str(newmcmc.maxSteps)+'.png',
                    facecolor='w',pad_inches=0.1)
        
        
        plt.close('all')
        print(ifile,file=sys.stderr)
        
        #plot optimal parameters versus best params------
        
        
def mainArgs(objpath,plotdir,nmodels,step):
    sys.argv = ['maintest.py', 
                '-objpath', str(objpath),
                '-plotdir', str(plotdir),
                '-nmodels',str(nmodels),
                '-stepEnsemble', str(step)]
    main()
    
if __name__ == "__main__":
    main()
    
