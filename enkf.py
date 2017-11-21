# -*- coding: utf-8 -*-
"""
Created on Thu Apr 06 11:47:07 2017

@author: kritt
"""
import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm 
def etkf(x,h,obs,R):
    """ETKF (Hunt et al. 2007, no localization)"""
    ens, ndim = x.shape; nobs = obs.shape[-1] 
    rho = 1.3
    xmean = x.mean(axis=0)
    xprime = (x - xmean).T
    ybmean = np.dot(h,xmean)
    ybprime = np.dot(h,xprime)
    C = np.dot(ybprime.T, inv(R))
    Pa_tilda = inv(((ens-1)*(np.eye(ens)))/rho+np.dot(C,ybprime))
    W_amean = sqrtm((ens-1)*Pa_tilda)
    W_abar = reduce(np.dot,[Pa_tilda,C, (obs-ybmean)])
    
#    W_a = W_amean+W_abar
    x_amean = (np.dot(xprime,W_abar).T+xmean)
    x_aprime = np.dot(xprime,W_amean).T
    Pa = reduce(np.dot,[xprime, Pa_tilda,xprime.T])
    x_a = x_amean  + x_aprime
#    return x_a, Pa    
    return x_a