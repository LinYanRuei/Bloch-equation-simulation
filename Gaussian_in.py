"""
Author Lin Yan Ruei @CCU Taiwan
"""

import numpy as np
import matplotlib.pyplot as plt
import my_function as fun
import hdf5Reader
from scipy.optimize import curve_fit as fit
from sys import float_info


#Define parameter
argDicts = {
    'att': 1.43353e2,   #dB
    'gain': 1.03127e2,  #dB
    'ke': 4.215e15,     
    'decoh': 1.143e6,   #Hz
    'relax': 2.232e6,   #Hz
    #Time
    'tau': 1,
    't0': 2.718,
    'span': 7.5,
    'dt': 0.001,
}

expDicts = {
    'pwr_range': np.linspace(-20, 20, 21),
    'detune_range': np.linspace(-3.6,3.6,37)
}

dirDicts = {
    #pwr_dep
    # 'datadir': r'C:\Users\user\OneDrive - 國立中正大學\桌面\job5\Data_sigma1us_Both4.766GHz_sweep_power\2022\07\Data_0720',
    # 'prefix': '\Chalmers_TD_timedelay_1us_pwr_',
    # 'suffix': 'dBm',
    # 'ext': '.hdf5',
    
    'datadir': r'C:\Users\user\OneDrive - 國立中正大學\桌面\job111\Data_sigma1us_Both4.766GHz_sweep_power2\2022\07\Data_0720',
    'prefix': '\Chalmers_TD_timedelay_1us_pwr_',
    'suffix': 'dBm',
    'ext': '.hdf5',
    
    #detune
    # 'datadir': r'C:\Users\user\OneDrive - 國立中正大學\桌面\job5\Data_sigma1us_probe4.766GHz_sweep_flux\2022\07\Data_0720',
    # 'prefix': '\Chalmers_TD_timedelay_1us_freq_tune_',
    # 'suffix': 'MHz',
    # 'ext': '.hdf5',

}

detune = 0          #Input_power
power = -20   
state = fun.g_state
state_save = []
q = fun.calculation_rk4(argDicts,power,detune,expDicts,dirDicts)
t_range = q.t_range
unit = q.r*2*np.pi*10/q.ke
sigX, sigY, sigZ = q.pauli_matrix(t_range,state,state_save)

input_chip = q.W_in(t_range)
input_V = input_chip*unit
input_V_digitizer = fun.dBm2vp(fun.vp2dBm(input_V)+q.att)    #Normal temperture
output_chip = q.W_out(sigX,sigY,t_range)
output_V = output_chip*unit                                  #Volt
output_digitizer = fun.dBm2vp(fun.vp2dBm(output_V)+q.gain)   #Normal temperture
output_BG = fun.dBm2vp(fun.vp2dBm(input_V)+q.gain)           #Normal temperture
t_range_use = t_range*1e-6/(q.r*2*np.pi*1e-6)

z_BG, x_BG, xname_BG = q.getData(True)     #BG
zz,xx,xxname = q.getData(False)      #EXP_OUT

plt.plot(t_range_use,output_digitizer)
plt.plot(xx,np.abs(zz[:,0]))
plt.show()
for i in range(len(q.pwr_range)):
    pwr = q.pwr_range[i]
    q = fun.calculation_rk4(argDicts,pwr,detune,expDicts,dirDicts)
    state_save=[]
    sigX,sigY,sigZ= q.pauli_matrix(t_range,state,state_save)             
    output_digitizer = fun.dBm2vp(fun.vp2dBm(q.W_out(sigX,sigY,t_range)*unit )+q.gain)
    plt.plot(t_range_use,output_digitizer,'--')
    plt.scatter(xx,np.abs(zz[:,i]),s=0.1,c ='red',marker= '.')
    plt.title(f'{pwr:.0f}dbm')
    plt.show()
    # plt.savefig(r'C:/Users/user/OneDrive - 國立中正大學/桌面/Plot_result/{}dBm.png'.format(pwr),dpi=300)
    # plt.clf()


data_range = np.arange(0,21,1)
t_plot = np.linspace(0,6,6981)*1e-6
############################################# Pwr_dep
result_raw_BG = []

q.plot_result(data_range,t_plot,z_BG,221)
q.plot_result(data_range,t_plot,zz,223)

pwr_range = np.linspace(-20,20,21)
result = []
for pwr in pwr_range:
    q = fun.calculation_rk4(argDicts,pwr,detune,expDicts,dirDicts)
    result.append(q.W_in(t_range)*unit)
result = np.array(result)
q.plot_result_2(t_range_use,pwr_range,result,222)

result = []
for pwr in pwr_range:
    q = fun.calculation_rk4(argDicts,pwr,detune,expDicts,dirDicts)
    state_save = []           
    sigX,sigY = q.pauli_matrix(t_range,state,state_save)             
    output_digitizer = fun.dBm2vp(fun.vp2dBm(q.W_out(sigX,sigY,t_range)*unit )+q.gain) 
    result.append(output_digitizer)
q.plot_result_2(t_range_use,pwr_range,result,224)
plt.show()
"""
detuning
q.plot_result(q.detune_range,t_plot,zz,223)

detune_range = np.linspace(-3.6,3.6,37)
result = []
for det in detune_range:
    q = fun.calculation_rk4(argDicts,power,det,expDicts,dirDicts)
    result.append(q.W_in(t_range)*unit)
result = np.array(result)
q.plot_result_2(t_range_use,detune_range,result,222)
q.plot_result_2(t_range_use,detune_range,result,221)

result = []
for det in detune_range:
    print(det)
    q = fun.calculation_rk4(argDicts,power,det,expDicts,dirDicts)          
    sigX,sigY = q.pauli_matrix(t_range,state,[])          
    output_digitizer = fun.dBm2vp(fun.vp2dBm(q.W_out(sigX,sigY,t_range)*unit)+q.gain) 
    result.append(output_digitizer)
q.plot_result_2(t_range_use,detune_range,result,224)
# plt.show()

"""




























