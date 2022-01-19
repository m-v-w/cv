# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 22:30:45 2021

@author: Oleg
"""

import numpy as np
import math
from sklearn.linear_model import Ridge
import scipy.special
#import hermite_functions as hf
import sklearn.metrics as sm
from sklearn.kernel_ridge import KernelRidge
import py_vollib_vectorized
import time
from tqdm import tqdm
import sys


def Euler1D(T,N,TrN,drift,diffusion, initX):
    #Simple Euler scheme for 1D SDE
    #parameters
    #T          time
    #N          number of timesteps (can be large)
    #TrN        number of trajectories (can be large)
    #drift      drift function of two parameters: time and x
    #diffusion  diffusion function of two parameters: time and x
    #initX      initial data
    
    start = time.perf_counter()
    delta=T/N
    sqrtdelta=math.sqrt(delta)

    result=initX+np.zeros(TrN)

    for i in tqdm(range(N)):
        curnoise=np.random.normal(0, 1, TrN)
        prev=result
        result=prev+drift(delta*i,prev)*delta+\
            np.multiply(diffusion(delta*i,prev),curnoise)*sqrtdelta    
    return result, time.perf_counter()-start
    
    
def EulerMVregr(T,N,TrN,driftX,h,sigma1,f,driftY,sigma2,initX,initY,corr,\
                method='rkhsfast', eps=0.01, lambd=0.00001, varkernel=5,\
                Nkernels=40,withconstant=False):
    #Euler scheme to solve 2D McKean Vlasov equation via regression
    #We are solving Equation (1.5) of Shkolnikov  arXiv:1905.06213
    
    #parameters
    #T          time
    #N          number of timesteps (should be small)
    #TrN        number of trajectories (should be very small especially for
    #           the kernel ridge regression method, otherwise can be larger)
    #driftX(time,x) aka b1 in Shkolnikov
    #h(time,x)  aka h in Shkolnikov
    #sigma1(time,x) aka sigma1 in Shkolnikov
    #f(time,x) aka f in Shkolnikov
    #driftY(time,x) aka b2 in Shkolnikov
    #sigma2(time,x) aka sigma2 in Shkolnikov 
    #initX      initial data in X
    #initY      initial data in Y
    #corr       correlation between BM in SDE for X and SDE for Y    
    #method     'hermite' for hermite polinomials, 'rkhs' for slow 'rkhs',
    #           'rkhsfast' for fasr rkhs
    #eps        parameter for the denominators to avoid division by 0
    #lambd      parameter for the Tikhonov regularizartion
    #varkernel  variance of the radial kernel for the RKHS method
    #Nkernels   number of kernels for faster ridge regression stuff
    
    start = time.perf_counter()
    delta=T/N
    sqrtdelta=math.sqrt(delta)
    
    barrho=np.sqrt(1-corr**2) # correlation stuff

    resultX=np.zeros(TrN)+initX   
    resultY=np.zeros(TrN)+initY

    for i in tqdm(range(N)):
        curnoiseX=np.random.normal(0, 1,  TrN)
        curnoiseY=corr*curnoiseX+ barrho*np.random.normal(0, 1, TrN)
        curtime=delta*i
        prevX=resultX
        prevY=resultY
        hY=h(curtime,prevY)# h(Y)
        fsqY=np.square(f(curtime,prevY))#f^2(Y)
        if method=='rkhs':
           zzz=1 
 #           estimatedrift,estimatediffusion=mlambdaKRR(prevX,prevY,lambd,varkernel)
        elif method=='rkhsfast':
            estimatedrift,estimatediffusion=mlambdaKRRfast(prevX,hY,fsqY,lambd,varkernel,Nkernels,withconstant)
        else:
            zzz=1
 #           estimatedrift,estimatediffusion=mlambdaHermite(prevX,prevY,lambd,lambd,Nbasisfunc)
        driftpart=driftX(curtime,prevX)*h(curtime,prevY)/(np.maximum(eps,estimatedrift))
        diffusionpart=sigma1(curtime,prevX)*f(curtime,prevY)/(np.sqrt(np.maximum(eps,estimatediffusion)))  
        resultX=prevX+ driftpart*delta+diffusionpart*curnoiseX*sqrtdelta 
        resultY=prevY+driftY(curtime,prevY)*delta+sigma2(curtime,prevY)*curnoiseY*sqrtdelta
    return resultX, time.perf_counter()-start


def mlambdaKRRfast(X,Y1,Y2,lambd,varkernel, Nkernels,withconstant=False):
    #calculates E(Y1|X) and E(Y2|X) via ridge regression in a faster way 
    #see https://mlweb.loria.fr/book/en/kernelridgeregression.html
    #we put f(\cdot)=sum alpha_j K(z_j, cdot)
    #and want to minimize 1/N \|Y_i -f(X_i)\| + lambda \|f\|_{RKHS}.
    #This problem can be solved explicitly:
    #alpha*=(K^T K + lambda *R)^{-1} K^T y
    #where R:=(K(z_i,z_j)), K=(K(z_j,x_i))
    
    #if we have constant, then its like having a special z=*, such that K(*)=1.
    #parameters:
    #lambd      parameter for the Tikhonov regularizartion
    #varkernel  variance of the radial kernel for the RKHS method
    #Nkernels   number of kernels for faster ridge regression stuff
    
    if (Nkernels==0) and (not withconstant):
        #throw an error
         sys.exit("0 kernels + no constant kernel = Not good!")
    
    N=len(X)    
    
    if (Nkernels==0) and withconstant:
        kern=np.ones((N,1))
        R=np.ones((1,1))
    else:        
        #first we generate an array of evenly distributed X: we take every len(X)/Mfast element of X
        Xreordered=np.sort(X)
        Xval=Xreordered[::math.ceil((len(Xreordered)+0.0)/Nkernels)]
 
        kern=radialkernel(X,Xval,varkernel)
        if withconstant:
            kern=np.hstack([np.ones((N,1)),kern])        
        
        if withconstant:
            R=np.ones((Nkernels+1,Nkernels+1))
            R[1:,1:]=radialkernel(Xval,Xval,varkernel)
        else:
            R=radialkernel(Xval,Xval,varkernel)
    
    kerntr=np.matrix.transpose(kern)
    coefmatrix=np.matmul(kerntr,kern)+N*lambd*R
    
    predictiondrift=np.matmul(kern,np.linalg.lstsq(coefmatrix,np.matmul(kerntr,Y1),rcond=None)[0])
    predictiondiffusion=np.matmul(kern,np.linalg.lstsq(coefmatrix,np.matmul(kerntr,Y2),rcond=None)[0])
    return predictiondrift,predictiondiffusion


def radialkernel(x,y,variance):
    #calculates kernel matrix K(x_i,y_j)
    xtwod=x[:,None]
    ytwod=y[:,None]
    varinv=1/(2*variance)
    res = np.exp(-varinv*sm.pairwise_distances(xtwod,ytwod))
    return res

def IVCalculation(S0,T,ST,Kmin=None,Kmax=None,Nstrikes=None):
    #Calculates implied vol.
    #We calculate E(S_T-K)_+ for S_T=X and Xhat.
    #We take 50 strikes in the interval (80%,120%)*K_0, where K_0 is the mean of Xhat
    #
    #Parameters
    #S0         initial price
    #T          time
    #ST         stock price at time T
    #Kmin       minimal strike
    #Kmax       maximal strike
    #Nstrikes   number of strikes 
    
    if Kmin is None:
        Kmin = 0.5*S0
    if Kmax is None:
        Kmax = 2*S0
    if Nstrikes is None:
        Nstrikes = 100
    K=np.linspace(Kmin,Kmax,Nstrikes)

    #Calculate implied volatility
    #calculate option price as E(S_T-K)_+ and then implied vol.
    distr=np.maximum(ST[np.newaxis,:] - K[:,np.newaxis],0)
    price=np.mean(distr,axis=1)
    result=py_vollib_vectorized.vectorized_implied_volatility(price, S0, K, T,\
            r=0, flag='c', q=0, on_error='warn', model='black_scholes_merton',return_as='numpy')
    return K,result


def MVCalculationmain (T,N,TrN,Nhat,TrNhat,sigmadupire,driftY,diffusionY,initX,initY,corr,\
                method='log', methodregr='rkhsfast', eps=0.01, lambd=0.00001, varkernel=5,\
                Nkernels=40):
    #parameters
    
    #T          time
    #N          number of timesteps (should be small)
    #TrN        number of trajectories (should be very small especially for
    #           the kernel ridge regression method, otherwise can be larger)
    #Nhat       number of timesteps for 1D Xhat calculations (can be large)
    #TrNhat     number of trajectories for 1D Xhat calculations (can be large)
    #sigmadupire(time,x) aka sigma_dupire in (1.12)-(1.13) of Shkolnikov
    #driftY(time,x) aka b2 in Shkolnikov
    #diffusionY(time,x) aka sigma2 in Shkolnikov 
    #initX      initial data in X
    #initY      initial data in Y
    #corr       correlation between BM in SDE for X and SDE for Y    
    #method     'log' for log increments, 'standard' otherwise
    #methodregr 'hermite' for hermite polinomials, 'rkhs' for slow 'rkhs',
    #           'rkhsfast' for fasr rkhs
    #eps        parameter for the denominators to avoid division by 0
    #lambd      parameter for the Tikhonov regularizartion
    #varkernel  variance of the radial kernel for the RKHS method
    #Nkernels   number of kernels for faster ridge regression stuff
    
    
    #Step0. Define correctly all the functions in (1.5) and (1.2)
    #       of Shkolikov depending whether we are in log or standard  mode.
    
    def b1(time,x):
        if  method=='log':
            res=-0.5*(sigmadupire(time, np.exp(x))**2)
        else:
            res=0*x
        return res
    
    def h(time,x):
        res=x
        return res
    
    def sigma1(time,x):
        if  method=='log':
            res=sigmadupire(time, np.exp(x))
        else:
            res=np.multiply(x,sigmadupire(time, x))
        return res
    
    def f(time,x):
        res=np.sqrt(np.maximum(x,eps))
        return res
    
    if method=='log':
        startX=np.log(initX)
    else:
        startX=initX
        
    #Step 1. Build 1D trajectories of \hat X. We have by Gyongy 
    #X\hat X = X in law so we can compare.
    
    result1d, time1d=Euler1D(T,Nhat,TrNhat,b1,sigma1,startX)
    #Step 2. Fun part. Build  the real X (and Y)
    
    result2d, time2d=EulerMVregr(T,N,TrN,b1,h,sigma1,f,driftY,diffusionY,startX,initY,corr,\
                methodregr, eps, lambd, varkernel,Nkernels)
        
    if method=='log':
        result1d=np.exp(result1d)
        result2d=np.exp(result2d)

    #Step 3. Calculate option prices and implied vol.
    
    S0=np.mean(initX)
     
    strikes, Xhatvols=IVCalculation(S0,T,result1d)
    strikes, Xvols=IVCalculation(S0,T,result2d)
    
    return result1d,result2d,strikes,Xhatvols,Xvols


#we don't need anything which is below
##############################################################################

# #functions for the Hermite method
# def basisfunc(x,Nbasis):
#     #calculates values of the basis functions at given x
#     #I took a constant + Hermite functions
    
#     result=np.transpose(np.vstack((np.zeros(len(x))+1,hf.hermite_functions(Nbasis,x))))
#     #result = np.reshape(np.zeros(len(x))+1,(-1,1)) # only 1 basis function!
#     #result=np.transpose(hf.hermite_functions(Nbasis,x))#without constant as a basis function!
#     return result

# def mlambdaHermite(X,Y,alphadrift,alphadiffusion,Nbasis):
#     #calculates E(h(Y)|X) and E(ff^2(Y)|X) via Tichonov regularization with Hermite functions as basis functions
#     #see https://mlweb.loria.fr/book/en/ridgeregression.html
    

#     #we are looking for f(x)=sum c_j e_j(x)
#     Fx=basisfunc(X,Nbasis)
#     N=len(X)
    
#     outcomedrift=hh(Y)
#     regdrift = Ridge(alpha=N*alphadrift,fit_intercept=False)
#     regdrift.fit(Fx,outcomedrift)
#     coefdrift=regdrift.coef_
#     predictiondrift=np.matmul(Fx,coefdrift)

#     outcomediffusion=np.square(ff(Y)) 
#     regdiffusion = Ridge(alpha=N*alphadiffusion,fit_intercept=False)
#     regdiffusion.fit(Fx,outcomediffusion)
#     coefdiffusion=regdiffusion.coef_
#     predictiondiffusion=np.matmul(Fx,coefdiffusion)
    
#     return predictiondrift,predictiondiffusion
    

# #functions for the kernel Ridge regression



# def mlambdaKRR(X,Y,alpharegr,kernelvariance):
#     #calculates E(h(Y)|X) and E(ff^2(Y)|X) via ridge regression 
#     #see https://mlweb.loria.fr/book/en/kernelridgeregression.html
    
#     N=len(X)
#     kern=radialkernel(X,X,kernelvariance)
    
#     outcomedrift=hh(Y)
#     regdrift = KernelRidge(alpha=N*alpharegr,kernel='precomputed')
#     regdrift.fit(kern,outcomedrift)
#     coefdrift=regdrift.dual_coef_
#     predictiondrift=np.matmul(kern,coefdrift)

#     outcomediffusion=np.square(ff(Y)) 
#     regdiffusion = KernelRidge(alpha=N*alpharegr,kernel='precomputed')
#     regdiffusion.fit(kern,outcomediffusion)
#     coefdiffusion=regdiffusion.dual_coef_
#     predictiondiffusion=np.matmul(kern,coefdiffusion)
    
#     return predictiondrift,predictiondiffusion






