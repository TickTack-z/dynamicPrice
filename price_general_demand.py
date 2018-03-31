# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 22:00:59 2017

@author: LeiXiao
"""
import numpy as np
import random
import scipy.stats
from scipy.optimize import minimize

def p6(price_historical,demand_historical,parameterdump=None):
    price_historical=np.array(price_historical)
    def MLE(beta,price_matrix,demand,D,wl):
        n=price_matrix.shape[0]
        binom_parameter=np.exp(beta[0]-beta[1]*price_matrix[0])/(1+sum(np.exp(beta[0]-beta[1]*price_matrix[i]) for i in range(n)))    
        llh=[np.log(scipy.stats.binom.pmf(demand[j],D,binom_parameter[j])) for j in range(len(demand))]
        llh=np.array(llh)
        llh[llh==-np.inf]=-999999
        return -sum(wl*llh)  

    #day 1
    if demand_historical.size==0:
        popt=0
        
    #day 2
    if demand_historical.size==1:
        n=price_historical.shape[0]
        #create possible_D
        possible_D=[]
        i=1
        while i<=(n+1)*demand_historical:
            if i>=demand_historical:
                possible_D.append(i)
            i=2*i
        parameterdump=possible_D
        otherprice_last=price_historical[:,0][1:]
        popt=max(min(otherprice_last),1)
            
    #day 3 to day 10
    if demand_historical.size>1 and demand_historical.size<10:
        t=demand_historical.size
        n=price_historical.shape[0]
        #check violated D and append new D
        possible_D=parameterdump
        for D in possible_D:
            if demand_historical[-1]>D:
                possible_D.remove(D)
        if len(possible_D)==0:   
            i=1
            while i<=(n+1)*demand_historical[-1]:
                if i>=demand_historical[-1]:
                    possible_D.append(i)
                    i=2*i
        #continue track others
        parameterdump=possible_D
        otherprice_last=price_historical[:,t-1][1:]
        popt=max(min(otherprice_last),1)
    
    #day 11
    if demand_historical.size==10:        
        t=demand_historical.size
        n=price_historical.shape[0]
        #check violated D and append new D
        possible_D=parameterdump
        for D in possible_D:
            if demand_historical[-1]>D:
                possible_D.remove(D)
        if len(possible_D)==0:   
            i=1
            while i<=(n+1)*demand_historical[-1]:
                if i>=demand_historical[-1]:
                    possible_D.append(i)
                    i=2*i
        #using EM algorithm to estimate the demand model
        demand=np.round(demand_historical)            
        price_mx=price_historical
        K=len(possible_D)
        theta=[1.0/K for i in range(K)]
        parameter=[]
        for k in range(K):
            parameter.append([0,1])        
        prob={}
        wl={}
        judge=False      
        while judge==False:
            for k in range(K):
                binom_param=np.exp(parameter[k][0]-parameter[k][1]*price_mx[0])/(1+sum(np.exp(parameter[k][0]-parameter[k][1]*price_mx[i]) for i in range(n)))
                prob[k]=np.array([scipy.stats.binom.pmf(demand[j],possible_D[k],binom_param[j]) for j in range(len(demand))])

            for k in range(K):
                wl[k]=prob[k]*theta[k]/sum(theta[j]*prob[j] for j in range(K))
                wl[k][np.isnan(wl[k])]=theta[k]
            parameter_new=[]  
            for k in range(K):
                parameter_new.append(minimize(MLE,parameter[k],args=(price_mx,demand,possible_D[k],wl[k]),bounds=((-5,None),(0.005+0.005*2**k,10)),method='L-BFGS-B').x)
            theta_new=[np.mean(wl[j]) for j in range(K)]
            temp=[np.linalg.norm(parameter_new[k]-parameter[k])<0.01 for k in range(K)]
            if sum(temp)==K:
                judge=True
            for k in range(K):
                parameter[k]=parameter_new[k]
            theta=theta_new
            
        #fit multivariate normal distribution and generate SAA sample
        price_mx=price_historical#change it for other day!!!!!
        sortedlist=[[] for i in range(n-1)]
        for j in range(t):
            temp=[price_mx[i][j] for i in range(n)[1:]]
            for i in range(n-1):
                sortedlist[i].append(np.sort(temp)[i])
        corr_matrix=np.cov(sortedlist)
        normal_mean=[np.mean(sortedlist[i]) for i in range(n-1)]
        saa=np.random.multivariate_normal(normal_mean,corr_matrix,1000).T            
        #find optimal price            
        option=np.linspace(0.1,100,num=999,endpoint=False)
        revenue=[]
        for i in range(999):
            rev=0
            for k in range(K):
                param=np.exp(parameter[k][0]-parameter[k][1]*option[i])/(1+np.exp(parameter[k][0]-parameter[k][1]*option[i])+sum(np.exp(parameter[k][0]-parameter[k][1]*saa[j]) for j in range(n-1)))
                demand=theta[k]*possible_D[k]*param
                rev+=sum(demand*option[i])
            revenue.append(rev)
        choice=np.argmax(revenue)
        popt=option[choice]
        parameterdump={'possible_D':possible_D,'parameter':parameter,'theta':theta}
    
    #day 12 and so on                
    if demand_historical.size>10:
        t=demand_historical.size
        n=price_historical.shape[0]
        #check violated D and append new D
        possible_D=parameterdump['possible_D']
        theta=parameterdump['theta']
        parameter=parameterdump['parameter'] 
        K=len(possible_D)
        index=[]
        change=0
        for k in range(K):
            if demand_historical[-1]>possible_D[k]:
                index.append(k)
        if len(index)>0:
            change=1
            for j in index:
                possible_D.pop(j)
                parameter.pop(j)
                
        if len(possible_D)==0:   
            i=1
            while i<=(n+1)*demand_historical[-1]:
                if i>=demand_historical[-1]:
                    possible_D.append(i)
                    parameter.append([0,1])
                    i=2*i
        K=len(possible_D)
        #theta=[1.0/K for i in range(K)]
        #parameter=[]
        #for k in range(K):
        #    parameter.append([0,1])  
        #using EM algorithm to estimate the demand model
        demand=np.round(demand_historical)            
        price_mx=price_historical 
        prob={}
        wl={}
        judge=False      
        while judge==False:
            for k in range(K):
                binom_param=np.exp(parameter[k][0]-parameter[k][1]*price_mx[0])/(1+sum(np.exp(parameter[k][0]-parameter[k][1]*price_mx[i]) for i in range(n)))
                prob[k]=np.array([scipy.stats.binom.pmf(demand[j],possible_D[k],binom_param[j]) for j in range(len(demand))])

            for k in range(K):
                wl[k]=prob[k]*theta[k]/sum(theta[j]*prob[j] for j in range(K))
                wl[k][np.isnan(wl[k])]=theta[k]
            parameter_new=[]  
            for k in range(K):
                parameter_new.append(minimize(MLE,parameter[k],args=(price_mx,demand,possible_D[k],wl[k]),bounds=((-5,None),(0.005+0.005*2**k,10)),method='L-BFGS-B').x)
            theta_new=[np.mean(wl[j]) for j in range(K)]
            temp=[np.linalg.norm(parameter_new[k]-parameter[k])<0.0001 for k in range(K)]
            if sum(temp)==K:
                judge=True
            for k in range(K):
                parameter[k]=parameter_new[k]
            theta=theta_new
        #fit multivariate normal distribution and generate SAA sample
        price_mx=price_historical[0:,(t-min(t,100)):]
        sortedlist=[[] for i in range(n-1)]
        for j in range(price_mx.shape[1]):
            temp=[price_mx[i][j] for i in range(n)[1:]]
            for i in range(n-1):
                sortedlist[i].append(np.sort(temp)[i])
        corr_matrix=np.cov(sortedlist)
        normal_mean=[np.mean(sortedlist[i]) for i in range(n-1)]
        saa=np.random.multivariate_normal(normal_mean,corr_matrix,1000).T            
        #find optimal price            
        option=np.linspace(0.1,100,num=999,endpoint=False)
        revenue=[]
        for i in range(999):
            rev=0
            for k in range(K):
                param=np.exp(parameter[k][0]-parameter[k][1]*option[i])/(1+np.exp(parameter[k][0]-parameter[k][1]*option[i])+sum(np.exp(parameter[k][0]-parameter[k][1]*saa[j]) for j in range(n-1)))
                demand=theta[k]*possible_D[k]*param
                rev+=sum(demand*option[i])
            revenue.append(rev)
        choice=np.argmax(revenue)
        popt=option[choice]
        parameterdump={'possible_D':possible_D,'parameter':parameter,'theta':theta}
                      
    return (popt, parameterdump)
