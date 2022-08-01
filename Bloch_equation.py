"""
Author Lin Yan Ruei @CCU Taiwan
"""

import matplotlib.pyplot as plt
import numpy as np
from qutip import *
import math
import sympy as sy

# Bloch equation
def bloch(p,steps,args,omega_in,delta):   
    x,y,z = p
    I = omega_in
    Q,r,gamma = args
    dx = -r*x - delta*y + I*z
    dy = delta*x - r*y - Q*z
    dz = -I*x + Q*y - gamma*z - gamma
    return [x+dx*steps,y+dy*steps, z+dz*steps]

def sigma_m(p,step,args,omega_in,delta):
    d = calculation(p,step,args,omega_in,delta)
    return d[0] - 1j*d[1]


def ref_coeff(p,step,args,omega_in,delta):
    r = 1 + 2*gamma/omega_in*sigma_m(p,step,args,omega_in,delta)
    return r

# Calculation
def calculation(p,step,args,omega_in,delta):
    for v in t:
        p = bloch(p,step,args,omega_in,delta)
        d.append(p)
    print(p)
    return d

# Parameter
omega_in = 1
gamma = 2
args = [0,1,gamma]
step = 0.001
t = np.arange(0,3,step)
delta = 0.
# Initial state
p0 = [0.,0.,1.]
p = p0
d = []
d = calculation(p,step,args,omega_in,delta)
dnp = np.array(d)


def f(t):
    return 2*np.exp(-2*t)-1

intrp = np.abs(2*np.exp(-1)-1-dnp[:,2])
print(intrp.argmin())
print(1/t[42])

# Plot the result
plt.plot(t, 1*dnp[:,0])
plt.title('bloch equation (x)')
plt.ylabel('expection value(x)')
plt.xlabel('Time')
# plt.show()
plt.plot(t, 1*dnp[:,1])
plt.title('bloch equation (y)')
plt.ylabel('expection value(y)')
plt.xlabel('Time')
# plt.show()
plt.plot(t, 1*dnp[:,2])
plt.title('bloch equation (z)')
plt.ylabel('expection value(z)')
plt.xlabel('Time')
plt.plot(t,f(t))
plt.show()


# s = (1000,1000)
# r = np.empty(s)
# for i in range(1000):
#     detuning = delta_range[i]
#     delta = detuning
#     for j in range(1000):
#         rf = rf_range[j]
#         r[i][j] = (np.abs(1-gamma*(1-1j*delta)/(1+delta**2+rf**2/gamma)))

# plt.imshow(r)
# plt.title('Analitical refrection coefficient')
# plt.ylabel('Rabi frequency')
# plt.xlabel('detuning')
# plt.colorbar()
# plt.show()
