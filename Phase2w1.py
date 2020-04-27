#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:37:25 2020

@author: Carmen
"""

########## SUN -VENUS TRANSFER ############
##########      WORKSHOP 1     ############


########## IMPORT PACKAGES ############
import numpy as np;
import sympy as sp;
from scipy.integrate import odeint;
from numpy import log as ln;
import time;
from math import *;
import random;
from sympy.solvers import solve;
from sympy.core.symbol import symbols;
import matplotlib.pyplot as plt;
import heapq


########## DEFINE CONSTANTS ############
G=6.6743015*10**(-11);                  #m3/kg/s2
m_sun=1.9891e+30;                       #kg
m_v=4.8685e+24;                         #kg
m_E=6.0477e+24;                         #kg
m_moon=7.342e+22 #kg  
mu_sun=G*m_sun;                         #m3/s2
a_v=108208179*10**3;                    #m
t_v=3089688;                            #s
t_E=325.25*86400;                       #s

############# FUNCTIONS ################


#Returns: colinear lagrange point coordinates (Newton's method)
#Input: x initial guess
def lagrangian(guess):
    
    #Convergence condition for position
    tolerance=10**(-15);  
    conv=1000;
    
    #Initial guess, assuming colinear
    x=guess;  y=0; z=0; 
    r1=sqrt((x+mu)**2);
    r2=sqrt((x-(1-mu))**2);
    
    while (conv>tolerance):

        f_mu=x-(1-mu)/r1**3*(x+mu)-mu/r2**3*(x-(1-mu));
        f_mu_prime= 3*(1-mu)*(mu+x)**2/r1**5-mu/r2**3-(1-mu)/r1**3-3*mu*(-mu-x+1)*(mu+x-1)/r2**5+1;
        
        #Compute convergence
        conv=abs(f_mu/f_mu_prime/x); 
        
        #Update satellite position
        x=x-f_mu/f_mu_prime;
        r1=abs(x+mu); 
        r2=abs(x-(1-mu)); 
    
    lagrange=[x,y,z];
    
    return lagrange


#Returns: Jacobian eigenvalues, eigenvectors
#Input: x,y,z vector
def stability(lagrange):
    
    #Assume planar displacement
    x,y=sp.symbols('x y')
    U=-((1-mu)/sp.sqrt((x+mu)**2+y**2)+mu/sp.sqrt((x-1+mu)**2+y**2))-0.5*(x**2+y**2);
    Uxx=sp.diff(sp.diff(U,x),x);
    Uxy=sp.diff(sp.diff(U,x),y);
    Uyy=sp.diff(sp.diff(U,y),y);
    
    #Take negative of result to account for absolute value r1, r2
    Uxx=-float(Uxx.subs([(x,lagrange[0]), (y, lagrange[1])]))
    Uxy=-float(Uxy.subs([(x,lagrange[0]), (y, lagrange[1])]))
    Uyy=-float(Uyy.subs([(x,lagrange[0]), (y, lagrange[1])]))
    
     #Define Jacobian
    A=[[0, 0, 1, 0],
       [0, 0, 0, 1],
       [Uxx, Uxy, 0, 2],
       [Uxy, Uyy, -2, 0]]
    
    #Find eigenvalues and eigenvectors
    eigenvalues=np.linalg.eig(A)[0];
    eigenvectors=np.linalg.eig(A)[1]; 
    
    return eigenvalues,eigenvectors


def manifold (lagrange,eigenvalue,eigenvector):
    
    per=10**(-5);   #perturbation, [-]
    t_max=5*t_E/t_v; #[-] #80
    dt=10**(-3);
    
    #Forward/backward integration
    if(eigenvalue>0):       #unstable, forward integrate
        t=np.arange(0,t_max,dt); 
        stability='Unstable'
        colour='r'
    else:                   #stable, backward integrate
        t=np.arange(t_max,0,-dt);
        stability='Stable'
        colour='g'
    
    def func(X, t, mu):
        
        x, y, vx, vy = X
        
        dXdt = [0,0,0,0]
        
        #Derivative
        dXdt[0]=vx
        dXdt[1]=vy
        dXdt[2]=2*vy+x-(1-mu)*(x+mu)/((x+mu)**2+y**2)**(3/2)-mu*(x-(1-mu))/((x+mu-1)**2+y**2)**(3/2)
        dXdt[3]=-2*vx+y-(1-mu)*y/((x+mu)**2+y**2)**(3/2)-mu*y/sqrt((x+mu-1)**2+y**2)**(3/2)
        
        return dXdt
    
    #Lagrangian state [x0,y0,vx0,vy0]
    X_0=[lagrange[0],lagrange[1],0,0]; 
    
    #Initial state
    X_i=X_0-per*(eigenvector/np.linalg.norm(eigenvector));
    
    sol = odeint(func, X_i, t, args=(mu,), atol=10**(-12),rtol=10**(-12)); 
    
    plt.plot(X_i[0],X_i[1], 'x'+colour , label='Initial point')
    plt.plot(sol[:, 0],  sol[:, 1], colour, label=stability)
    
    
    return sol


#Returns: real eigenvalues, corresponding eigenvectors, direction arrays
#Input: lagrange
def real(lagrange):
    
    #Identify real eigenvalues and corresponding eigenvector
    i=0;
    real_eigenvalues=[];
    real_eigenvectors=[];
    eigenvalues,eigenvectors=stability(lagrange);
    
    while (i<len(eigenvalues)):
        if abs(eigenvalues[i].imag)<(10**(-15)):
            real_eigenvalues=np.append(real_eigenvalues, eigenvalues[i].real)
            if (len(real_eigenvectors)==0):
                real_eigenvectors=eigenvectors[:,i].real
            else:
                real_eigenvectors=[real_eigenvectors, eigenvectors[:,i].real]
        i=i+1;
    
    return real_eigenvalues, real_eigenvectors





############### MAIN ###################
ratio=m_sun/m_v;                        #m1/m2
#ratio=m_E/m_moon;
mu=1/(ratio+1);                         #CR3BP normalization

#Initial and target Lagrange
L_start_dl=lagrangian(-1.000001)          #L3, [-]
L_end_dl=lagrangian(1.003)     #1.009371)   #1.003             #L2, [-]

L_start=np.multiply(L_start_dl,a_v)+[mu*a_v,0,0];    #L3, [m] wrt Sun
L_end=np.multiply(L_end_dl,a_v)+[mu*a_v,0,0];        #L2, [m] wrt Sun

#ver1=abs(L_start[0]-107201002*1000)/10720100/1000*100    #Validation error L1 [%]
#ver2=abs(L_end[0]- 109223156*1000)/109223156/1000*100    #Validation error L2 [%]

#Stability of Lagrange
stability_L_start=stability(L_start_dl)[0];
stability_L_end=stability(L_end_dl)[0];

#Manifolds
    #Manifolds for initial Lagrange
eigenvalues,eigenvectors=real(L_start_dl)
manifold(L_start_dl,eigenvalues[0],eigenvectors[0])
manifold(L_start_dl,eigenvalues[1],eigenvectors[1])

#Display
plt.plot(0,0,'oy',label='Sun')
plt.plot(L_start_dl[0],L_start_dl[1],'om', label='L3' ) 
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()
    
    #Manifolds for target Lagrange
eigenvalues,eigenvectors=real(L_end_dl)
manifold(L_end_dl,eigenvalues[0],eigenvectors[0])
manifold(L_end_dl,eigenvalues[1],eigenvectors[1])

#GDisplay
plt.plot(0,0,'oy',label='Sun')
plt.plot(L_end_dl[0],L_end_dl[1],'oc', label='L2' ) 
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()
