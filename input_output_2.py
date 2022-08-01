"""
Author Lin Yan Ruei @CCU Taiwan
"""

import matplotlib.pyplot as plt
import numpy as np
from qutip import *
import math
import sympy as sy
import scipy.integrate as integrate
import scipy.interpolate as interpld
import os
# import fittingModels as fm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Bloch equation
def bloch(p,steps,args,omega_in,delta,v,tau):   
    x,y,z = p
    I = W_in(v,tau,omega_in)   #Omega_in
    Q,r,gamma = args
    dx = -r*x - delta*y + I*z
    dy = delta*x - r*y - Q*z
    dz = -I*x + Q*y - gamma*z - gamma
    return [x+dx*steps,y+dy*steps, z+dz*steps]

# Calculation
def calculation(p,steps,args,omega_in,delta,tau):
    c = []
    for v in t:
        p = bloch(p,steps,args,omega_in,delta,v,tau)
        c.append(p)
    return c

def W_out(x,y,tau,omega_in,t):
    return np.abs(W_in(t,tau,omega_in)+gamma*(x-y*1j))


def sigma(p,steps,args,omega_in,delta,tau):
    return np.array(calculation(p,steps,args,omega_in,delta,tau))

def W_in(t,tau,omega_in):
    # return omega_in*np.exp(t/tau)*(np.heaviside(-t, 0))  exponential
    return omega_in*np.exp(-((t-15)/np.sqrt(2)*tau)**2)

def efficiency(tau,omega_in):
    cw = []
    x = []
    y = []
    cw = calculation(p,steps,args,omega_in,delta,tau)
    cw = np.array(cw)
    x = cw[:, 0]
    y = cw[:, 1]
    t0 = np.argmax(W_out(x,y,tau,omega_in,t))
    q = W_out(x,y,tau,omega_in,t)
    # t0 = np.argmax(W_out(x,y,tau,omega_in,t))
    # q = W_out(x,y,tau,omega_in,t)
    rabiout_s = integrate.simps((q[t0:]**2), t[t0:])
    rabiin_s = integrate.simps((W_in(t,tau,omega_in))**2,t)
    rate_simps=rabiout_s/rabiin_s
    return rate_simps
# Parameter

omega_in = 0.1
gamma = 2
delta = 0
r = 1
args = [0,r,gamma]
steps = 0.001
t = np.arange(0,30,steps)
step = 0.01
# Initial state
p0 = [0.,0.,-1.]
p = p0
d = []
c = []
tau = 1

k = np.array(sigma(p,steps,args,omega_in,delta,tau))
sigX = k[:, 0]
sigY = k[:, 1]
sigZ = k[:, 2]


# t0 = np.argmax(W_out(sigX,sigY,tau,omega_in,t))
# c = []
# c = W_out(sigX,sigY,tau,omega_in,t)
t0 = np.argmax(W_out(sigX,sigY,tau,omega_in,t))
c = []
c = W_out(sigX,sigY,tau,omega_in,t)

p1 = plt.plot(t,W_in(t,tau,omega_in),label = 'Input_pulse')
p2 = plt.plot(t,W_out(sigX,sigY,tau,omega_in,t),label = 'Output_pulse')
plt.title('Input & output pulse')
plt.legend(loc = 'upper right')
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.show()
print(efficiency(tau,omega_in))

result = []
rf_range = np.arange(0.01,3.01,step)
tau_range = np.arange(0.2,0.5,step/10)
print(len(rf_range),len(tau_range),len(k))
print(sigX)

result_line = []
for j in range(len(tau_range)):
    rf = 0.1
    tau = tau_range[j]
    result_line.append(efficiency(tau,rf))

plt.plot(tau_range,result_line)
plt.show()
for i in range(len(rf_range)):
    result.append([])
    rf = rf_range[i]
    print(i)
    for j in range(len(tau_range)):
        tau = tau_range[j]
        result[i].append(efficiency(tau,rf))



result = np.array(result)
X,Y = np.meshgrid(tau_range, rf_range)
cp = plt.contourf(X,Y,result,100,cmap='jet')

cbar = plt.colorbar(cp)
cbar.ax.set_ylabel('efficiency')
plt.title('Efficiency in different tau and rabi frequency')
plt.ylabel('Rabi frequency')
plt.xlabel('tau')
plt.show()
