# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 17:32:18 2017

@author: kriti Bhargava
Main code that imports the class for Lorenz 1996 model
and the module for data assimilation 
and does ETKF for Lorenz 1996 model

Some parts of this code are based on a similar code by jswhit on :
https://github.com/jswhit/L96/blob/master/L96ensrf.py
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.linalg import qr
from scipy.linalg import orth
from L96 import L96
import enkf
import Growth_vectors as GV
im = 40 # number of dimensions for Lorenz 96 model
F = 8  # Forcing for the L96 model for chaotic system
dt =0.01 # time step corresponding to 72 minutes or 1.2 hours
tsteps =6000 # number of steps
spinup = 300 # number of steps for spinup
ens = im+1  # N-1= n where N is the number of ensemble members and n is dimension of system
obs_stdev = 0.1
num_obs = im
R = np.eye(num_obs)*obs_stdev**2 # Observation Error Covariance
num_da_cycle = 600
da_inter = tsteps/num_da_cycle # steps in a single forecast-assimilation interval
mx =[] # for saving model results
enx = [] # for saving ensemble forecast results
ena = [] # for saving ensemble analysis
fcsterr = []
time = np.arange(0,tsteps*dt, dt)# for saving timestamps
plot = 1
x_plot_truth = np.zeros((num_da_cycle,im))
obs_plot = np.zeros((num_da_cycle,num_obs))
time_plot = np.zeros((num_da_cycle))


""" 

-----------------GENERATING TRUTH AND CALCULATING VARIANCE---------------------

"""

model = L96(n=im, F=F, dt=dt) # model object
#spinup
for i in range(spinup):
    model.advance()
# Truth run        
for i in range(tsteps):
    model.advance()
    mx.append(model.x)

xtruth =np.array(mx,np.float)

 

""" 

-----------------GENERATING OBSERVATIONS AND OBS. VARIANCE---------------------

"""
H = (np.eye(num_obs))
rs = np.random.RandomState() # selecting a random seed for observations
obs = np.empty(xtruth.shape, xtruth.dtype)
for i in range(xtruth.shape[0]):
    obs[i] = np.dot(H,xtruth[i])
obs = obs + obs_stdev*rs.standard_normal(size=obs.shape)

""" 

-----------------GENERATING INITIAL ENSEMBLE AND SPINNING UP-------------------

"""

enx=[]
ensemble =L96(mem=ens, n=im, F=F,dt=dt)
initial = ensemble.x
for i in range(spinup):
    ensemble.advance()
ensemble.advance()
enx.append(ensemble.x)
 #forecast from the updated initial condition
""" 
#
#---------------------DATA ASSIMILATION CYCLE BEGINS HERE-----------------------
#
#"""
#
n=0
for n_assim in range(0,tsteps,da_inter):
    fcsterr.append(ensemble.x-xtruth[n_assim])
    x = enkf.etkf(ensemble.x, H, obs[n_assim,:], R)
    ensemble.x = x
    ena.append(ensemble.x)
    for i in range(da_inter):
        ensemble.advance()
    enx.append(ensemble.x)
#    creating variabLes to plot at times of assimilation
    x_plot_truth[n,:] = xtruth[n_assim,:]
    obs_plot[n,:] = obs[n_assim,:]
    time_plot[n] = time[n_assim]
    n+=1
x_fcst = np.array(enx,np.float)
x_anal = np.array(ena,np.float)
x_ferr = np.array(fcsterr,np.float)
x_anal_mean = x_anal.mean(axis=1)
x_fcst_mean = x_anal.mean(axis=1)

""" 
#
#---------------------CALCULATING GROWTH VECTORS-----------------------
#
#
"""
# Calculating Singular Vectors
TLM_pre = GV.TLM(x_plot_truth[-2],im,dt) # TLM for the preceeding time period
TLM_sub = GV.TLM(x_plot_truth[-1],im,dt) # TLM for the subsequent time period
u_req,s, v = np.linalg.svd(TLM_pre, full_matrices=1)
u,s,v_req = np.linalg.svd(TLM_sub, full_matrices=1)

# Calculating Forward and Backward Lyapunov Vectors by computing TLM over 100 
# time steps before and after
k = 500 # no of steps to calculate LVs
mx2=[]
for i in range(0,k):
    mx2.append(model.x)
    model.advance()
xtruth_f = np.array(mx2)    
TLM_minus = np.eye(im)
TLM_plus = np.eye(im)
H = np.random.randn(im, im)
Q1, R1 = qr(H)
Q2, R2 = qr(H)
for i in range (0,k):
# Using QR decomposition    
    TLM_minus=GV.TLM(xtruth[tsteps-k-1+i,:],im,dt)
    V1= np.dot (TLM_minus,Q1)    
    Q1,R1 =qr(V1)
    
    TLM_plus=GV.TLM(xtruth_f[i,:],im,dt)
    V2=np.dot (TLM_plus.T,Q2)
    Q2,R2 =qr(V2)   


# Using TLM only

 
#    minus = GV.TLM(xtruth[tsteps-k-1+i,:],im,dt)
#    TLM_minus= np.dot(minus,TLM_minus)
#    plus=GV.TLM(xtruth_f[i,:],im,dt)
#    TLM_plus = np.dot(plus,TLM_plus)
#   
#
#u,s1,FLV = np.linalg.svd(TLM_plus, full_matrices=1) 
#BLV,s2,v = np.linalg.svd(TLM_minus, full_matrices=1)  
#lambda1 = np.log(s1)/(k*dt)
BLV = Q1
FLV = Q2

# Computing Covariant Lyapunov Vector
CLV = np.zeros((im,im))
for k in range(0,im):
    Lf = FLV[:,np.arange(k,im)]
    Lb = -1*BLV[:,np.arange(0,k+1)]
    A=np.column_stack((Lf,Lb))
    Y=GV.null(A)
    Y_v = Y[0:im-k,0]
    g1=np.dot(Lf,Y_v)
    g1_norm = np.linalg.norm(g1)
    CLV[:,k]=g1/g1_norm
    
""" 
#
#---------------------PROJECTING ERRORS ON GROWTH VECTORS-----------------------
#
#
"""
error = np.array(fcsterr[-1])
LSV_proj = np.log10(np.absolute(np.dot(u_req.T,error.T)))
RSV_proj = np.log10(np.absolute(np.dot(v_req.T,error.T)))
FLV_proj = np.log10(np.absolute(np.dot(FLV.T,error.T)))
BLV_proj = np.log10(np.absolute(np.dot(BLV.T,error.T)))
CLV1 = np.dot(inv(np.dot(CLV.T,CLV)), CLV.T)
CLV_proj = np.log10(np.absolute(np.dot(CLV1,error.T)))

""" 
#
#---------------------PLOTTING-----------------------
#
#
"""
fig = plt.figure(figsize=(10, 40))
plt.subplot(5, 1, 1)
plt.imshow(LSV_proj, cmap = 'jet',vmin=-4.5,vmax=0)
plt.title('(a) Projection of error replicates on L SV of TLM')
plt.colorbar(orientation = 'vertical')
plt.ylabel('L SV index')

plt.subplot(5, 1, 2)
plt.imshow(RSV_proj, cmap = 'jet',vmin=-4.5,vmax=0)
plt.colorbar(orientation = 'vertical')
plt.title('(b) Projection of error replicates on R SV of TLM')
plt.ylabel('R SV index')

plt.subplot(5, 1, 3)
plt.imshow(FLV_proj, cmap = 'jet',vmin=-4.5,vmax=0)
plt.colorbar(orientation = 'vertical')
plt.title('(c) Projection of error replicates on fwd LV')
plt.ylabel('fwd LV index')

plt.subplot(5, 1, 4)
plt.imshow(BLV_proj, cmap = 'jet',vmin=-4.5,vmax=0)
plt.colorbar(orientation = 'vertical')
plt.title('(d) Projection of error replicates on bwd LV')
plt.ylabel('bwd LV index')

plt.subplot(5, 1, 5)
plt.imshow(CLV_proj, cmap = 'jet',vmin=-3.5,vmax=1)
plt.colorbar(orientation = 'vertical')
plt.title('(e) Projection of error replicates on covariant LV')
plt.ylabel('bwd LV index')

"""
#
#---------------------EXTRA stuff-----------------------

if (plot):
    plt.figure(figsize=(18, 6), dpi=200)
    plt.subplot(111)
    plt.xlabel( 'Time' )  #x name
    plt.ylabel( 'x20' )  #y name
    plt.xlim(0.0, dt*tsteps) #x limit
    plt.ylim(-15.0, 15.0) #y limit
    black = plt.plot(time_plot[:],x_plot_truth[:,19], color="black", linewidth=1, label= "Truth")
    red = plt.plot(time_plot[:],x_fcst_mean[:,19], color="red", linewidth=1, label = "Forecast")
    blue = plt.plot(time_plot[:],x_anal_mean[:,19], color="blue", linewidth=1, label = "Analysis")
    green = plt.plot(time_plot[:],obs_plot[:,19], color="green", linewidth=1, label = "Observations" )
    plt.legend()
"""    