"""
Author Lin Yan Ruei @CCU Taiwan
"""

import numpy as np
from sys import float_info
from hdf5Reader import hdf5Handle
import matplotlib.pyplot as plt

g_state = np.array([0,0,-1])   #ground state
e_state = np.array([0,0,1] )   #excited state

# unit conversion functions (from fittingModels by Alaster)
def p2dBm(p):  # W to dBm
    p[p <= 0.] = float_info.min
    return 10 * np.log10(p / 1e-3)

def vp2p(vp, Z0=50):  # Vp to W
    return vp**2 / 2 / Z0

def vp2dBm(vp, Z0=50):  # Vp to dBm
    return p2dBm(vp2p(vp, Z0))

def dBm2p(dBm):  # dBm to W
    return 1e-3 * 10 ** (dBm/10)

def p2vp(p, Z0=50):  # W to Vp
    return np.sqrt(2 * Z0 * p)

def dBm2vp(dBm, Z0=50):  # dBm to Vp
    return p2vp(dBm2p(dBm), Z0)

class calculation_rk4:
    def __init__(self,argDicts,amp,detune,expDicts,dirDicts):
        self.amp = amp
        self.att = argDicts['att']
        self.gain = argDicts['gain']
        self.ke = argDicts['ke']
        self.r = argDicts['decoh']
        self.deco = argDicts['relax']/argDicts['decoh']  #cancel dim
        self.delta = detune/argDicts['decoh']
        unit = self.r*2*np.pi*1e-6
        self.t0 = argDicts['t0']*unit
        self.tau = argDicts['tau']*unit
        self.dt = argDicts['dt']*unit
        self.t_range = np.linspace(0,7,6981)*unit
        self.pwr_range = expDicts['pwr_range']
        self.datadir = dirDicts['datadir']
        self.prefix = dirDicts['prefix']
        self.suffix = dirDicts['suffix']
        self.ext = dirDicts['ext']
        self.detune_range = expDicts['detune_range']

    def W_in(self,t):
        G_Vp = dBm2vp(self.amp-self.att)*np.exp(-(t-self.t0)**2/(2*self.tau**2))  # gaussian in (voltage)
        Rabi_in = self.ke*np.sqrt(vp2p(G_Vp))/(self.r*2*np.pi) # no unit
        return Rabi_in
    # dX/dt = AX+B
    def A(self,t): return np.array(
    [[-1, -1*self.delta, self.W_in(t)], [self.delta, -1, 0], [-1*self.W_in(t), 0, -1*self.deco]])  # d=detune
    
    def dy(self,X, t):
        return np.dot(self.A(t), X.T)+(np.array([0, 0, -self.deco]).T)
    
    def rk4(self,state,t):
        k1 = self.dy(state, t)
        k2 = self.dy(state+self.dt/2.0*k1.T, t+self.dt/2.0)
        k3 = self.dy(state+self.dt/2.0*k2.T, t+self.dt/2.0)
        k4 = self.dy(state+self.dt*k3.T, t+self.dt)
        state = state+self.dt*(k1.T+2.0*k2.T+2.0*k3.T+k4.T)/6.0
        t = t+self.dt
        return state, t
    
    def W_out(self,x,y,t):
        return np.abs(self.W_in(t)+self.deco*(x-1j*y))

    def imag(self, x, y, t):
        return np.imag(self.W_in(t)+self.deco*(x-y*1j))

    def real(self, x, y, t):
        return np.real(self.W_in(t)+self.deco*(x-y*1j))

    def getData(self,bg=True):
        flag = ''
        if bg:
            flag = '_BG'
        for i, power in enumerate(self.pwr_range):
            filename = self.datadir + self.prefix + f'{power:.0f}' + self.suffix + flag + self.ext
            data = hdf5Handle('Time', 's', file=filename, log_ch_idx=2)
            data_chB = hdf5Handle('Time', 's', file=filename, log_ch_idx=3)
            data.slice()
            data_chB.slice()
            if i == 0:
                x, _, z, xname, _, _, _ = data.output()
                zMat = 1j * np.zeros((len(x), len(self.pwr_range)))
            else:
                z = data.output()[2]
            z_chB = data_chB.output()[2]
            zMat[:, i] = z[:, 0] / np.exp(1j * np.angle(z_chB[:, 0]))

        return zMat, x, xname
    
    def fit_func(self,x_,a,b,x0):
        return a*np.exp(-1*(x_-x0)**2/2*b**2)

    def plot_result(self,data_range,t_range,z,k):
        b = []
        for i in range(len(data_range)):
            b.append(z[:,i])
        b = np.array(b)
        plt.subplot(k)
        X,Y = np.meshgrid(t_range,data_range)
        cp_raw_BG = plt.contourf(X,Y,b,1000,cmap='jet')
        cbar_raw_BG = plt.colorbar(cp_raw_BG)
        cbar_raw_BG.ax.set_ylabel('amplitude')
        plt.title('Raw data')
        plt.ylabel('Input (power)')
        plt.xlabel('time (s)')
        # plt.show()

    def plot_result_2(self,x,y,result,k):
        plt.subplot(k)
        X,Y = np.meshgrid(x,y)
        cp_fit_in = plt.contourf(X,Y,result,1000,cmap='jet')
        cbar_fit_in = plt.colorbar(cp_fit_in)
        cbar_fit_in.ax.set_ylabel('amplitude')
        plt.title('simulation')
        plt.ylabel('Input (power)')
        plt.xlabel('time (s)')
        # plt.show()

    def pauli_matrix(self,t_range,state,state_save):
        for i in range(len(t_range)):
            t = t_range[i]
            state,t = self.rk4(state,t)
            state_save.append(state)
        state_save = np.array(state_save)
        sigX = state_save[:,0]
        sigY = state_save[:,1]  
        sigZ = state_save[:,2]
        return sigX, sigY, sigZ

class bloch_equation:
    def __init__(self,states,steps,args,t,I,k):
        self.state = states
        self.steps = steps
        self.Q = args['Q']
        self.r = args['r']
        self.gamma = args['gamma']
        self.delta = args['detuning']
        self.pl = args['pl']
        self.t = t
        self.I = I
        self.k = []

    def bloch(self,states,steps):   
        x,y,z = states
        dx = -self.r*x - self.delta*y + self.I*z
        dy = self.delta*x - self.r*y - self.Q*z
        dz = -self.I*x + self.Q*y - self.gamma*z - self.gamma
        return [x+dx*steps,y+dy*steps, z+dz*steps]
    # Calculation
    def calculation(self,states,t):
        for v in t:
            states = self.bloch(states,self.steps)
            self.k.append(states)
        return self.k
    
    def plt(self,t,array,pl):
        plt.plot(t,array[:,pl])
        plt.title('bloch equation ')
        plt.ylabel('expection value')
        plt.xlabel('Time')
        plt.show()
    
