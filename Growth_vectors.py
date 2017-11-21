# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 14:35:29 2017

@author: kritt
"""

import numpy as np
def TLM(x,n,dt=0.01):
#    n = np.size(x)
    Jacob = np.zeros((n,n))
    for i in range (0,n):
        Jacob[i,(i-2)%n]=-x[(i-1)%n]
        Jacob[i,(i-1)%n]=x[(i+1)%n]-x[(i-2)%n]
        Jacob[i,(i)%n]=-1
        Jacob[i,(i+1)%n]=x[(i-1)%n]
#    x_TLM = Jacob*dt +np.eye(n)        
    x_TLM = Jacob*dt +np.eye(n)
    return x_TLM

def null(a, rtol=1e-5):
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return v[rank:].T.copy()