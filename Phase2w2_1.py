#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:48:57 2020

@author: Carmen
"""

########## SUN -VENUS TRANSFER ############
##########     WORKSHOP 2      ############


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
from mpl_toolkits import mplot3d;
import heapq

from Phase2w1 import lagrangian
from Phase2w1 import stability 
from Phase2w1 import manifold
from Phase2w1 import real

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
sigma_crit=1.53;                        #g/m2     

############# FUNCTIONS ################

#Input: dimensionless lagrange point, shift (1x3 arrays)
#Returns: required lightness number beta (float), feasability (bool)
def performance(lagrange,shift):
    
    #Compute artificial Lagrange
    if (lagrange[0]>0):
        L_art=np.subtract(lagrange,shift)
    else:
        L_art=np.add(lagrange,shift)
        
    #L_art=[0.95,0,0.1]
    
    #Effective potential differential
    x,y,z=sp.symbols('x y z')
    U=-((1-mu)/sp.sqrt((x+mu)**2+y**2+z**2)+mu/sp.sqrt((x-1+mu)**2+y**2+z**2))-0.5*(x**2+y**2)
    Ux=sp.diff(U,x)
    Ux=float(Ux.subs([(x,L_art[0]), (y, L_art[1]), (z, L_art[2])])); 
    Uy=sp.diff(U,y)
    Uy=float(Uy.subs([(x,L_art[0]), (y, L_art[1]), (z, L_art[2])])); 
    Uz=sp.diff(U,z)
    Uz=float(Uz.subs([(x,L_art[0]), (y, L_art[1]), (z, L_art[2])])); 
    Udiff=[Ux, Uy, Uz]; #print('U_Diff', Udiff)
    
    #Compute required solar sail orientation
    n_unit=Udiff/np.linalg.norm(Udiff); 
    
    r1=[L_art[0]+mu, L_art[1],L_art[2]]
    r2=[L_art[0]-(1-mu), L_art[1],L_art[2]]
    r1_unit=r1/np.linalg.norm(r1); #print('r1 unit', r1_unit)

    #Compute solar sail lightness nmbr
    beta=np.linalg.norm(r1)**2*np.dot(Udiff,n_unit)/((1-mu)*(np.dot(r1_unit,n_unit))**2)
    
    #Check feasability
    #https://engineering.purdue.edu/people/kathleen.howell.1/Publications/Masters/2000_McInnes.pdf
    if (np.dot(r1_unit,np.multiply((-1),Udiff))>=0):
        feasable=False
    else:
        feasable=True

    return beta, feasable


#Returns: solar sail area (float, m2)
#Input:solar sail lighness nmbr (float)
def ssarea(beta):
    
    A=mass*1000*beta/sigma_crit
    
    return A


def ssmanifold (lagrange,eigenvalue,eigenvector,alpha):
    
    #print('eig', eigenvalue)
    per=10**(-5);                               #perturbation, [-]
    t_max=5*t_E/t_v;                            #[-] #80
    dt=10**(-3);
    beta=0.03263836397273173                    #[-]
    alpha=np.radians(alpha)                     #[rad]
    delta=0                                     #[rad]
    
    #Forward/backward integration
    if(eigenvalue>0):       #unstable, forward integrate
        t=np.arange(0,t_max,dt); 
        stability='Unstable'
        stab=False
        colour='r'
    else:                   #stable, backward integrate
        t=np.arange(t_max,0,-dt);
        stability='Stable'
        stab=True
        colour='g'
    
    #print('t',t)
    
    def func(X, t, mu, beta, alpha, delta):
        
        x, y, z, vx, vy, vz = X
        r1=[x+mu,y,z]
        r1_unit=r1/np.linalg.norm(r1);
        z_unit=[0,0,1]
        
        n=np.vstack((cos(alpha), sin(alpha)*sin(delta), sin(alpha)*cos(delta))); #print('n',n)
        theta=np.cross(z_unit,r1_unit); #print('theta',theta)
        eta=np.cross(r1_unit,theta); #print('eta',eta)
        rot=np.vstack((r1_unit,theta,eta)); 
        n_unit=rot.dot(n)
        n_unit=n_unit/np.linalg.norm(n_unit); #print('rot', rot, 'nun', n_unit)
        acc_ss=beta*(1-mu)/np.linalg.norm(r1)**2*(np.dot(r1_unit,n_unit))**2*n_unit
        #print('acc_ss', acc_ss)
        
        dXdt = [0,0,0,0,0,0]
        
        #Derivative
        dXdt[0]=vx
        dXdt[1]=vy
        dXdt[2]=vz
        dXdt[3]=acc_ss[0]+2*vy+x-(1-mu)*(x+mu)/((x+mu)**2+y**2)**(3/2)-mu*(x-(1-mu))/((x+mu-1)**2+y**2)**(3/2)
        dXdt[4]=acc_ss[1]-2*vx+y-(1-mu)*y/((x+mu)**2+y**2)**(3/2)-mu*y/sqrt((x+mu-1)**2+y**2)**(3/2)
        dXdt[5]=acc_ss[2]
        
        return dXdt
    
    #Lagrangian state [x0,y0,vx0,vy0]
    X_0=[lagrange[0],lagrange[1],0,0]; 
    
    #Initial state, now 3D with zero z, vz
    X_i=X_0+per*(eigenvector/np.linalg.norm(eigenvector));
    X_i=np.append(X_i,0)
    X_i=np.insert(X_i, 2, 0); #print ('Initial state', X_i)
    
    sol = odeint(func, X_i, t, args=(mu,beta,alpha,delta,), atol=10**(-12),rtol=10**(-12)); 
    #print('sol', sol)

    return sol



############### MAIN ###################
ratio=m_sun/m_v                         #m1/m2
#ratio=m_sun/m_E
mu=1/(ratio+1)                          #CR3BP normalization
mass=10                                 #[kg]

#Initial and target Lagrange
L_start_dl=lagrangian(-1.000001)          #L3, [-]
L_end_dl=lagrangian(1.003)                #L2, [-]

#Artificial Lagrange, Solar sail performance number
shift=[0.011,0,0]
Beta_start, feasable_start = performance(L_start_dl, shift)
Beta_end, feasable_end = performance(L_end_dl, shift)

#Solar sail length
length_start=sqrt(ssarea(Beta_start))
length_end=sqrt(ssarea(Beta_end)); print (length_start, length_end)

#Solar Sail Manifolds
alpha=np.arange(-70,70,2.5)
i=0
minimum=[]
best_alphas=[]
best_minimums=[]
best_l3in=[]
best_l2in=[]
#ANN_init=[]
#ANN_target=[]
#error=[]


while (i<len(alpha)):

    #Manifolds for (classical) initial Lagrange
    #Unstable for initial Lagrange (L3)
    eigenvalues,eigenvectors=real(L_start_dl)
    if (eigenvalues[0]>0):
        initial = ssmanifold(L_start_dl,eigenvalues[0],eigenvectors[0],alpha[i]) 
    else:
        initial = ssmanifold(L_start_dl,eigenvalues[1],eigenvectors[1],alpha[i])
    
    #Manifolds for (classical) target Lagrange
    #Stable for target Lagrange (L2)
    eigenvalues,eigenvectors=real(L_end_dl)
    if (eigenvalues[0]<0):
        target = ssmanifold(L_end_dl,eigenvalues[0],eigenvectors[0],alpha[i])
    else:
        target = ssmanifold(L_end_dl,eigenvalues[1],eigenvectors[1],alpha[i])

    
    #Euclidean error computation
    m=0
    minimum=1000
    while (m<len(initial)):
        #Check difference one L3 - all L2 and find minimum
        difference=np.linalg.norm(np.subtract(target[:],initial[m][:]),axis=1)
        if (minimum>np.min(difference)):
            minimum=np.min(difference)
            l3_index=m
            l2_index=np.argmin(difference)
            l_minimums=[initial[m],target[l2_index]]
            alpha_min=alpha[i]
        #Iterate for all L3
        m=m+1
    
    best_alphas=np.append(best_alphas, alpha_min)
    best_minimums=np.append(best_minimums, minimum)
    best_l3in=np.append(best_l3in,l3_index)
    best_l2in=np.append(best_l2in,l2_index)
    
    #Save initial time, target time and corresponding error
    #ANN_init=np.append(ANN_init, t_initial)
    #ANN_target=np.append(ANN_target, t_target)
    #error=np.append(error,np.min(difference))
    
    #Iteration
    i=i+1

#Minimum Euclidean-norm error as a function of the cone angle
plt.plot(best_alphas,best_minimums,'-')
plt.xlabel('Cone angle (deg)')
plt.ylabel('Euclidean error (norm) [-]')
plt.grid()
plt.show()

#Best cone angle, absolute min Euclidean
print('best cone:', best_alphas[np.argmin(best_minimums)])
print('Euclidean norm:', min(best_minimums))
initial_stop=int(best_l3in[np.argmin(best_minimums)])
target_stop=int(best_l2in[np.argmin(best_minimums)])

#Time of orbit
tt=time_un=np.arange(0, 5*t_E/t_v , 0.001)
tt_initial=tt[initial_stop]*t_v/t_E
tt_target=tt[target_stop]*t_v/t_E
print('Time from L3 (Ey):', tt_initial)
print('Time to L2 (Ey):', tt_target)

#Error in position, velocity [km, km/s]
pos_err=np.linalg.norm(initial[initial_stop][:3]-target[target_stop][:3])
pos_err=pos_err*0.723*149597870.7
vel_err=np.linalg.norm(initial[initial_stop][-3:]-target[target_stop][-3:])
vel_err=vel_err*0.723*149597870.7/t_v

ax = plt.axes(projection='3d')
ax.plot3D(initial[:initial_stop, 0],  initial[:initial_stop, 1], initial[:initial_stop,2], 'r', label='Unstable')
ax.plot3D(target[:target_stop, 0],  target[:target_stop, 1], target[:target_stop,2], 'g', label='Stable')
#ax.scatter3D(X_i[0],X_i[1],X_i[2], 'x'+colour , label='Initial point')
    
#Display
ax.scatter3D(0,0,0,'oy',label='Sun')
ax.scatter3D(L_end_dl[0],L_end_dl[1], L_end_dl[2],'om', label='L2' ) 
ax.scatter3D(L_start_dl[0],L_start_dl[1], L_start_dl[2],'om', label='L3' ) 
plt.legend(loc='best')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title('Solar sail trajectory')
plt.grid()
plt.show()
    



