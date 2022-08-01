import matplotlib.pyplot as plt
import numpy as np
from qutip import *
import math
import sympy as sy
import scipy.integrate as integrate
import scipy.interpolate as interpld

def bloch(p,steps,args,omega_in,delta,v,tau):   
    x,y,z = p
    I = W_in(v,tau,omega_in)   #Omega_in
    Q,r,gamma = args
    dx = -r*x - delta*y + I*z
    dy = delta*x - r*y - Q*z
    dz = -I*x + Q*y - gamma*z - gamma
    return [x+dx*steps,y+dy*steps, z+dz*steps]

def calculation(p,steps,args,omega_in,delta,tau,t):
    for v in t:
        p = bloch(p,steps,args,omega_in,delta,v,tau)
        c.append(p)
    return c

def W_out(x,y,tau,omega_in,gamma,t):
    return np.abs(W_in(t,tau,omega_in)+gamma*(x-y*1j))

def sigma(p,steps,args,omega_in,delta,tau):
    k = np.array(calculation(p,steps,args,omega_in,delta,tau))
    return k

def W_in(t,tau,omega_in):
    return omega_in*np.exp(t/tau)*(np.heaviside(-t, 0)) 

c = []