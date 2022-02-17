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

def main():
    p=argparse.ArgumentParser(description="filename for plotting")
    p.add_argument("-objpath",type=str,help="name file or dir \
                                           for loading mcmc object")
    p.add_argument("-plotdir",type=str,help="name dir for saving plots")
    p.add_argument("-initmodel",type=int,help="initial model for \
                                                   plotting")
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
        if newmcmc.totalSteps<= args.initmodel:
            ensemble=newmcmc.samples[int((newmcmc.totalSteps/newmcmc.thin_by-1)/2)::args.stepEnsemble,:,:]
            xaxis=np.arange(int(newmcmc.totalSteps/2),newmcmc.totalSteps+1,args.stepEnsemble*newmcmc.thin_by)
            nmodels=len(xaxis)
            logprob=newmcmc.logprob[int((newmcmc.totalSteps/newmcmc.thin_by-1)/2)::args.stepEnsemble,:]
        else:
            ensemble=newmcmc.samples[int(args.initmodel/newmcmc.thin_by-1)::args.stepEnsemble,:,:]
            xaxis=np.arange(args.initmodel,newmcmc.totalSteps+1,args.stepEnsemble*newmcmc.thin_by)
            nmodels=len(xaxis) 
            logprob=newmcmc.logprob[int(args.initmodel/newmcmc.thin_by-1)::args.stepEnsemble,:]
    
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
        plt.plot(autoxaxis[np.nonzero(newmcmc.autocorr)],50*newmcmc.autocorr[np.nonzero(newmcmc.autocorr)],label=r'50 * $\tau$ estimate')
        plt.xlabel('Iteration')
        ax=plt.gca()
        ax.legend()
        
        
        #create folder for saving figure
        if not os.path.exists(args.plotdir+'figures/'+'autocorr/'):
            os.makedirs(args.plotdir+'figures/'+'autocorr/')
            
        plt.savefig(args.plotdir+'figures/'+'autocorr/'
                    +newmcmc.modelName+'_'+str(newmcmc.maxSteps)+'.pdf',
                    facecolor='w',pad_inches=0.1)
        
        #lag, acc rate and y per time for each model ----------------------------
        #indxlagparams=paramsList.index(lagparamsList[0])
        
        retreatt=np.zeros((nmodels*newmcmc.nwalkers,len(newmcmc.tr.accuModel._times)))
        lagt=np.zeros((nmodels*newmcmc.nwalkers,len(newmcmc.tr.accuModel._times)))
        acct=np.zeros((nmodels*newmcmc.nwalkers,len(newmcmc.tr.accuModel._times)))
        tmpt=np.zeros((nmodels*newmcmc.nwalkers,len(newmcmc.tr.accuModel._times),2))
        
        indxw=0
        for i in range(0,nmodels):
            for w in range(0,newmcmc.nwalkers):
                iparams=dict(zip(newmcmc.tr.all_parameter_names,ensemble[i,w,:]))
                newmcmc.tr.set_model(iparams)
                
                retreati=newmcmc.tr.retreat_model_t
                lagti=newmcmc.tr.lagModel.get_lag_at_t(newmcmc.tr.accuModel._times)
                accti=newmcmc.tr.accuModel.get_accumulation_at_t(newmcmc.tr.accuModel._times)
                tmpti=np.array(newmcmc.tr.get_trajectory(newmcmc.tr.accuModel._times))
                
                retreatt[indxw]=retreati
                lagt[indxw]=lagti
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
        plt.subplot(3,1,1)
        plt.plot(timesub/1000000,retreatt[:,0::subsample].T*1000,c="gray",
                                            alpha=0.1, zorder=-1)
        plt.plot(timesub/1000000,retreatt[indxbest,0::subsample]*1000,c="b")
        plt.xticks([], [])
        plt.title('R(L(t),t) (mm/year)')
        
        #plot acct
        plt.subplot(3,1,2)
        plt.plot(timesub/1000000,1000*acct[:,0::subsample].T,c="gray",
                                            alpha=0.1, zorder=-1)
        plt.plot(timesub/1000000,1000*acct[indxbest,0::subsample],c="b")
        plt.title('A(t) (mm/year)')
        plt.xlabel('Time (Myr)')
        
        if '_Obliquity_' in newmcmc.modelName:
            #plot obliquity data
            data,times =  load_obliquity_data()
            titledata = 'Obliquity (deg)'
        else:
            #plot insolation data
            data,times =  load_insolation_data()
            titledata = 'Insolation (W/m^2)'
        
        plt.subplot(3,1,3)
        plt.plot(-1*times/1000000,data)
        plt.title(titledata)
        plt.xlabel('Time (Myr)')
        
        #create folder for saving figure
        if not os.path.exists(args.plotdir+'figures/'+'ar_rates/'):
            os.makedirs(args.plotdir+'figures/'+'ar_rates/')
            
        plt.savefig(args.plotdir+'figures/'+'ar_rates/'
                    +newmcmc.modelName+'_'+str(newmcmc.maxSteps)+'.png',
                    facecolor='w',pad_inches=0.1)
        
        # tmp fit ---------------------------------------------------
        plt.figure()
        bestTMP=tmpt[indxbest,:,:]
        plt.plot(bestTMP[:,0],bestTMP[:,1],c='b',label='Best TMP')
        
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
        
        for i in range(nmodels*newmcmc.nwalkers):
            indx=i
            plt.plot(tmpt[indx,:,0],tmpt[indx,:,1],c="gray", alpha=0.1, zorder=-1)
        plt.plot(tmpt[indx,:,0],tmpt[indx,:,1],c="gray", alpha=0.1, zorder=-1,label='Ensemble models')
        plt.xlabel("Horizontal dist [m]")
        plt.ylabel("V. dist [m]")
        ax=plt.gca()
        ax.legend(bbox_to_anchor=(0.5, -0.3), loc='upper left')
        #ymin,ymax=ax.get_ylim()
        #xmin,xmax=ax.get_xlim()
        xmax=np.max(newmcmc.xdata)+1000
        ymin=np.min(newmcmc.ydata)-100
        ax.set_ylim(ymin,0)
        ax.set_xlim(0,xmax)
        ax.set_box_aspect(ratioyx)
        
        #plot times on upper axis
        ax2=ax.twiny()
        color='m'
        ax2.set_xlabel('Time before present ( Million years)',color=color)
        plt.scatter(xnear,ynear,marker="o",color='m')
        plt.errorbar(x=newmcmc.xdata, xerr=xerr, y=newmcmc.ydata, yerr=yerr, 
                 c='r', marker='.', ls='',label='Observed TMP')
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
        plt.errorbar(x=newmcmc.xdata, xerr=xerr, y=newmcmc.ydata, yerr=yerr, 
                 c='r', marker='.', ls='')
        
         #create folder for saving figure
        if not os.path.exists(args.plotdir+'figures/'+'tmpFull/'):
            os.makedirs(args.plotdir+'figures/'+'tmpFull/')
    
            
        plt.savefig(args.plotdir+'figures/'+'tmpFull/'
                    +newmcmc.modelName+'_'+str(newmcmc.maxSteps)+'.png',
                    facecolor='w',pad_inches=0.1)
        
        #plot variances as histograms-----------------------------------
        stdsx=errorbar1d*newmcmc.tr.meters_per_pixel[0]
        stdsy=errorbar1d*newmcmc.tr.meters_per_pixel[1]
        
        plt.figure()
        plt.subplot(1,2,1)
        plt.hist(stdsx,bins=100)
        plt.axvline(x=stdsx[indxbest],color='k',label='std best model',
                    linestyle='dashed')
        plt.title('Horizontal loc')
        plt.xlabel('std (m^2)')
        plt.ylabel('# models')
        plt.legend()
        
        plt.subplot(1,2,2)
        plt.hist(stdsy,bins=100)
        plt.axvline(x=stdsy[indxbest],color='k',label='std best model',
                    linestyle='dashed')
        plt.yticks([], [])
        plt.title('Vertical loc')
        plt.xlabel('std (m^2)')
        plt.legend()
        
        #create folder for saving figure
        if not os.path.exists(args.plotdir+'figures/'+'var/'):
            os.makedirs(args.plotdir+'figures/'+'var/')
    
            
        plt.savefig(args.plotdir+'figures/'+'var/'
                    +newmcmc.modelName+'_'+str(newmcmc.maxSteps)+'.pdf',
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
            
        plt.figure()
        plt.hist(ages,bins=100)
        plt.axvline(x=ages[indxbest],color='k',label='Age best model',
                    linestyle='dashed')
        plt.xlabel('Age (Myr)')
        plt.ylabel('# models')
        plt.legend()
        
        #create folder for saving figure
        if not os.path.exists(args.plotdir+'figures/'+'age/'):
            os.makedirs(args.plotdir+'figures/'+'age/')
    
        plt.savefig(args.plotdir+'figures/'+'age/'
                    +newmcmc.modelName+'_'+str(newmcmc.maxSteps)+'.pdf',
                    facecolor='w',pad_inches=0.1)
        
        plt.close('all')
        print(ifile,file=sys.stderr)
        
        #plot optimal parameters versus best params------
        
        
def mainArgs(objpath,plotdir,init,step):
    sys.argv = ['maintest.py', 
                '-objpath', str(objpath),
                '-plotdir', str(plotdir),
                '-initmodel',str(init),
                '-stepEnsemble', str(step)]
    main()
    
if __name__ == "__main__":
    main()
    
