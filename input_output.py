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

def sigma_m(p,steps,args,omega_in,delta):
    d = calculation(p,steps,args,omega_in,delta)
    return (d[0] - 1j*d[1])


def ref_coeff(p,steps,args,omega_in,delta):
    r = 1 + (gamma/omega_in)*sigma_m(p,steps,args,omega_in,delta)
    return np.abs(r)

# Calculation
def calculation(p,steps,args,omega_in,delta):
    for v in t:
        p = bloch(p,steps,args,omega_in,delta)
        # c.append(p)
    return p

def r_ana(gamma,delta,rf):
    return np.abs(1-gamma*(1-1j*delta)/(1+delta**2+rf**2/gamma))
# Parameter
omega_in = 0.01
gamma = 1.5
args = [0,1,gamma]
steps = 0.01
t = np.arange(0,10,steps)
step = 0.01
# Initial state
p0 = [0.,0.,-1.]
p = p0
d = []
c = []
delta = 0
print(sigma_m(p,steps,args,omega_in,delta))
print(ref_coeff(p,steps,args,omega_in,delta))
################################################ Numerical
delta_range = np.arange(-10,10,steps)
rf_range = np.arange(0.2,10.2,steps/2)

r = []
d_range = np.linspace(0,10,11)
# plt.subplot(2, 2, 1) 
for detuning in d_range:
    delta = detuning
    for rf in rf_range:
        r.append(ref_coeff(p,steps,args,rf,delta))
    plt.plot(rf_range,r,label = detuning)
    plt.title('Numerical refrection coefficient')
    plt.ylabel('reflection coefficient')
    plt.xlabel('Rabi frequency')
    r = []
plt.text(9,0.92,'rabi_freq',fontsize = 10)

plt.legend(loc='right')
plt.show()
# plt.subplot(2, 2, 2) 
###############################################
result_n = []
for i in range(len(delta_range)):
    result_n.append([])
    delta = delta_range[i]
    print(i)
    for j in range(len(rf_range)):
        rf = rf_range[j]
        result_n[i].append(ref_coeff(p,steps,args,rf,delta))
result_n = np.array(result_n)
X,Y = np.meshgrid(delta_range, rf_range)
cp = plt.contourf(X,Y,result_n.T,100,cmap='jet')
plt.yscale('log')
cbar = plt.colorbar(cp)
cbar.ax.set_ylabel('reflection coefficient amplitude')
plt.title('Numerical refrection coefficient')
plt.ylabel('Rabi frequency')
plt.xlabel('detuning')
plt.show()

# ################################################ Analitical
delta_range = np.arange(-10,10,steps)
rf_range = np.arange(0.2,10.2,steps/2)
delta_range = np.array(delta_range)
rf_range = np.array(rf_range)
r = []
d_range = np.linspace(0,10,11)
# plt.subplot(2, 2, 3) 
for detuning in d_range:
    delta = detuning
    for rf in rf_range:
        r.append (r_ana(gamma,delta,rf))
    plt.plot(rf_range,r,label = detuning)
    r = []
plt.text(9,0.85,'detuning',fontsize = 10)
plt.legend(loc='right')
plt.title('Analitical refrection coefficient in different detuning')
plt.xlabel('Rabi frequency')
plt.ylabel('refection coeff')
plt.show()
# ####################################################
# plt.subplot(2, 2, 4) 
result_a = []
for i in range(len(delta_range)):
    result_a.append([])
    delta = delta_range[i]
    for j in range(len(rf_range)):
        rf = rf_range[j]
        result_a[i].append(r_ana(gamma,delta,rf))
result_a = np.array(result_a)
X,Y = np.meshgrid(delta_range, rf_range)
cp = plt.contourf(X,Y,result_a.T,100,cmap='jet')
cbar = plt.colorbar(cp)
plt.title('Analitical refrection coefficient')
plt.yscale('log')
plt.ylabel('Rabi frequency')
plt.xlabel('detuning')
cbar.ax.set_ylabel('reflection coefficient amplitude')
plt.show()
# ###############




