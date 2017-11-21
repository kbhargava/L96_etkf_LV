# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 17:14:57 2017

@author: Kriti Bhargava

This code is for initializing and running Lorenz 1996 model.
The code is based on a similar code by jswhit on :
https://github.com/jswhit/L96/blob/master/L96.py

"""

import numpy as np

class L96:

    def __init__(self,mem=1,n=40,dt=0.01,F=8):
        self.n = n
        self.dt = dt   
        self.F = F
        self.mem = mem
        if self.mem ==1:
            self.x = np.loadtxt('C:\Users\kritt\OneDrive - umd.edu\Academics\Courses\Graduate\Initial.txt')
        else:
            rs = np.random.RandomState()
            self.x =F+np.loadtxt('C:\Users\kritt\OneDrive - umd.edu\Academics\Courses\Graduate\Initial.txt')+0.1*rs.standard_normal(size=(mem,n))

    def f(self):
        dxdt = np.empty_like(self.x)
        x=self.x
        n=self.n
        for j in range (n):
            if ((self.mem)==1 ):
                dxdt[:][j]=(x[:][(j+1)%n]-x[:][(j-2)%n])*x[:][(j-1)%n]-x[:][j]+self.F
            else :
                for i in range (self.mem):
                    dxdt[i][j]=(x[i][(j+1)%n]-x[i][(j-2)%n])*x[i][(j-1)%n]-x[i][j]+self.F
            # % sign here ensures the circular nature of the Lorenz 96 model
        return dxdt

    def advance(self):
        x_old = self.x
        k1 = self.f()
        self.x = x_old +k1*self.dt/2.0
        k2 = self.f()
        self.x = x_old+k2*self.dt/2.0
        k3 = self.f()
        self.x = x_old+k3*self.dt
        k4 = self.f()
        x_new = x_old + (self.dt/6)*(k1+2.0*k2+2.0*k3+k4)
        self.x = x_new
