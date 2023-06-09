#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 10:48:50 2023

@author: hkerino
"""

import numpy as np
import matplotlib.pyplot as plt
import george
from george import kernels
from astropy.timeseries import LombScargle
import random 
from scipy.stats import loguniform, gamma, uniform
import time 

import warnings
warnings.filterwarnings("ignore")


#%%
#functions 

def lorentzian_components(params_gr, t):

    '''
    Simulated effects of pulsations and granulations
    
    source: Dumusque et al 2011
    https://www.aanda.org/articles/aa/pdf/2011/01/aa14097-10.pdf
    
    Inputs
        - params_gr := A1, A2, A3, B1, B2, B3, C1, C2, C3, AL, Gm, nu0, cste
        - t = time in seconds

    ''' 
    
    # define frequency arrays  
    # The frequency ν goes from 1/T to the Nyquist frequency in steps of 1/T
    N    = len(t)
    Ttot = (max(t)-min(t)) # total time in seconds
    dt = np.median(np.diff(t)) # sampling rate
    freq_Nyq = (1.0/dt)/2 # Nyquist frequency
    freq  = np.arange(1.0/Ttot,freq_Nyq,1.0/Ttot, dtype='float64')
    # freq  = np.linspace(1.0/Ttot, 1./2/dt, len(t)) # Hz

    # define the power spectra as a sum of 3 components for granulation, mesogranulation and supergranulation + oscillations
    A1, A2, A3, B1, B2, B3, C1, C2, C3, AL, Gm, nu0, cste = params_gr 
    
    VPSD = A1 / (1+(B1*freq)**C1) + A2 / (1+(B2*freq)**C2) +  A3 / (1+(B3*freq)**C3) + \
                AL * (Gm**2/((freq-nu0)**2+Gm**2)) +  cste  
    # VPSD /= 1e6     # units of VPSD is (m/s)**2/Hz

           
    # take random phase between 0 and 2pi
    phase = 2*np.pi*np.random.rand(1,len(VPSD))
    #print(len(phase), len(VPSD))
    
    # Synthetic radial velocity measurements
    ysyn = np.zeros(N)
    for i in range(N):
        ysyn[i] = np.sum(np.sqrt(VPSD)*np.sin(2*np.pi*t[i]*freq+phase))
       
    return ysyn


def generate_regular_data_H0(params_gr, params_act, N, t):
    
    '''
    Inputs:
    - params_gr:= 
            see function lorentzian_components(params_gr, t)
    - params_act:=
            - amplitude (GP amplitude)
            - gam = 2/epsilon in the paper (GP gamma)
            - logP = log(P) in the paper (GP period)
            - met = 1.0 in the paper (GP metric)
    
    - N = number of data points
        
    Outputs:
        - t = time (days)
        - y = RV (m/s)
        - yerr =  intrinsic errors (m/s)
    '''
    
    # =========================================================================
    # Generate a Lorentzian component that represents the pulsations and
    # three components for the granulation, mesogranulation and su-
    # pergranulation
    ts    = t*24*3600 # seconds
    ygr = lorentzian_components(params_gr, ts)
    
    # =========================================================================
    # Generate activity component based on GP
    
    # define the QP kernel for the GP (see https://george.readthedocs.io/en/latest/user/kernels/)
    amp, gam, logP, met  = params_act
    k  = kernels.ExpSine2Kernel(gamma=gam, log_period=logP)
    k *= kernels.ExpSquaredKernel(metric=met) # metric correspondind to r^2/lambda  is 1
    k *= amp**2 
    # print(k.get_parameter_vector())
    
    gp = george.GP(k)

    # generate synthetic dataset
    yact = gp.sample(t)
    
    # =========================================================================
    # Generate the intrincsic errors
    sig = 0.30 # m/s
    yerr = np.random.normal(loc=0, scale=sig, size=N) # this has to be defined according to the paper
    #print('std(yerr) = %f m/s'%np.std(yerr))

    # =========================================================================
    # Generate the final synthetic time series
    # activity + granulation + the intrinsic errors 
    ytot = ygr + yact + yerr * np.random.randn(N)
    
    return ytot, yerr


def generate_planete(params_pl,t):
    """
    - params_pl
            - Ppl (Orbital period)
            - K (semi-amplitude)
            - T0 ( Phase à l'origine)
    """
    Ppl, K, T0 = params_pl
    return K*np.sin(2*np.pi*(t-T0)/Ppl)


#%%
#Over-sampling
Ttot = 200 # days
facteur_surech = 10
N = Ttot*facteur_surech +1 # number of data point in a regularly sampled grid
t = np.linspace(0,Ttot,N) #days
dt = Ttot*facteur_surech/(N-1) #After sampling


# Irregular sampling 1 : with normal distribution
facteur_ech = 10
t_ir = np.zeros(Ttot)

ind0 = round( 0 + facteur_ech*np.abs(np.random.normal(0, 0.1)))
t_ir[0] = t[ ind0 ]
ind_fin = round( (N-1) - facteur_ech*np.abs(np.random.normal(0, 0.1)))
t_ir[-1] = t[ ind_fin ]

pas = N/Ttot
for i in range(Ttot-2):
    ind = round((i+1)*pas+facteur_ech*np.random.normal(0, 0.1))
    t_ir[i+1] =  t[ind]
    
fmin = 1/Ttot
fmax = (1/dt)/2
freq = np.arange(fmin,fmax,fmin/10)

#%%
#Parameters for time series
## Grannulation + Oscillation
A1, A2, A3  = 0.027, 0.003, 0.3*1e-3 # m/s
B1, B2, B3  = 7.4*3600, 1.2*3600, 17.9*60 # seconds
C1, C2, C3  = 3.1, 3.9, 8.9 # dimensionless
AL, Gm, nu0 = 2.6*1e-3, 0.36*1e-3, 2.4*1e-3 # (m/s), Hz, Hz
cste        = 1.4e-4 # (m/s)**2/Hz

params_gr = [A1, A2, A3, B1, B2, B3, C1, C2, C3, AL, Gm, nu0, cste]

##Activity signal
Prot = 62 # days -- Prot to be chosen randomly in HARPS sample (see paper)

amp  = gamma.rvs(2.0, 0.5) 
epsilon = uniform.rvs(0.5, 1)
tau = np.random.normal(3*Prot, 0.1*Prot) 
gam  = 2.0/epsilon #not change
logP = np.log(Prot) #not change 
met  = 1.0 # not change 

params_act = [amp, gam, logP, met]


#%%
#Generated time series
y_ir, yerr_ir = generate_regular_data_H0(params_gr, params_act, len(t_ir), t_ir)
    
#%%
#Add 0,1,2,3 and 4 planets for each star 
time_series = np.zeros((Ttot,5))
LS = np.zeros((len(freq),5))
freq_pl = np.zeros(5)

time_series[:,0] = y_ir #Without planet
LS[:,0] = LombScargle(t_ir, y_ir,normalization='standard').power(freq,method='cython')#Without planet

for i in range(1,5):
    Ppl = random.uniform(10*np.min(np.diff(t_ir)),Ttot/2)
    K = loguniform.rvs(0.1, 10)
    T0 =  random.uniform(0,Ppl)
    params_pl = [Ppl,K,T0]
    freq_pl[i] = 1/Ppl
    
    y_pl = generate_planete(params_pl, t_ir)
    time_series[:,i] = y_pl + time_series[:,i-1]
    LS[:,i] = LombScargle( t_ir, time_series[:,i],normalization='standard').power(freq,method='cython')
    
    
    
#%%
plt.close("all")

plt.figure(1)
plt.suptitle("Time series and their LS for different numbers of planets injected")
plt.subplot(321), plt.scatter(t_ir,time_series[:,0],s=5,c='r'), plt.plot(t_ir,time_series[:,0],label = 'No plantet',c='k'),plt.legend()
plt.subplot(323), plt.scatter(t_ir,time_series[:,1],s=5,c='r'), plt.plot(t_ir,time_series[:,1],label = '1 plantet',c='k'),plt.legend()
plt.subplot(325), plt.scatter(t_ir,time_series[:,4],s=5,c='r'), plt.plot(t_ir,time_series[:,4],label = '4 plantets',c='k'),plt.legend()
plt.xlabel("time (d)")
plt.ylabel("Radial velocity (m/s)")

plt.subplot(322),plt.plot(freq, LS[:,0], label = 'No plantet'), plt.legend()
plt.subplot(324),plt.plot(freq, LS[:,1], label = '1 plantet'),plt.legend(), plt.plot([freq_pl[1],freq_pl[1]] , [min(LS[:,1]),max(LS[:,1])],'r--')
plt.subplot(326)
plt.plot(freq, LS[:,4],label = '4 plantets'), plt.legend()
plt.plot([freq_pl[1],freq_pl[1]] , [min(LS[:,4]),max(LS[:,4])],'r--')
plt.plot([freq_pl[2],freq_pl[2]] , [min(LS[:,4]),max(LS[:,4])],'g--')
plt.plot([freq_pl[3],freq_pl[3]] , [min(LS[:,4]),max(LS[:,4])],'b--')
plt.plot([freq_pl[4],freq_pl[4]] , [min(LS[:,4]),max(LS[:,4])],'y--')
plt.xlabel("frequency (d⁻¹)")
plt.ylabel("Power")


#%%
#For each periodogram, remove highest frequency and generate a new one x4
y = time_series[:,1]
p = LS[:,1]
f = freq_pl[1]

best_freq = freq[np.argmax(p)]
y_fit = LombScargle(t_ir, y).model(t_ir,best_freq)
y_res = y - y_fit 

new_p = LombScargle(t_ir, y_res)
best_freq_new = freq[np.argmax(new_p)]
y_fit_new = LombScargle(t_ir, y_res).model(t_ir,best_freq_new)
y_res_new = y_res - y_fit_new

plt.figure(2)
plt.subplot(311)
plt.scatter(t_ir, y, label='time serie',s=5,c='k'),plt.plot(t_ir,y_fit,label='best fit'), plt.legend()
plt.subplot(312)
plt.scatter(t_ir,y_res, label='Time series with best frequency removed',s=5,c='k'),plt.legend()
plt.subplot(313)
plt.scatter(t_ir, y_res, label='time serie',s=5,c='k'),plt.plot(t_ir,y_fit_new,label='new best fit'), plt.legend()


#%%
def generate_4_periodogrammes(t,f, y):
    """
    Parameters
    ----------
    t : time (day)
    f : frequency ( day⁻1)
    y : original time serie

    Returns
    -------
    4 times series with their lomb-scargle 
    """
    y_t = np.zeros((len(t), 4))
    LS  = np.zeros((len(f), 4))
    y_t[:,0] = y 
    
    for i in range(3):
        LS[:,i] = LombScargle(t, y_t[:,i]).power(f,method='cython')
        best_freq = f[np.argmax(LS[:,i])]
        y_fit = LombScargle(t, y_t[:,i]).model(t,best_freq)
        y_t[:,i+1] = y_t[:,i] - y_fit
       
    LS[:,3] = LombScargle(t, y_t[:,3]).power(f,method='cython')
    
    return y_t, LS



Ppl = random.uniform(10*np.min(np.diff(t)),Ttot/2)
K = loguniform.rvs(0.1, 10)
T0 =  random.uniform(0,Ppl)
params_pl = [Ppl,K,T0]

y_activity, y_err = generate_regular_data_H0(params_gr, params_act, len(t_ir), t_ir)
y_pl = generate_planete(params_pl, t_ir)
y = y_activity + y_pl
series, LS_p = generate_4_periodogrammes(t_ir, freq, y)

plt.figure(3)
plt.suptitle("Time series with their Lomb-Scargle periodogramme generated by removing the best frequency fit")
plt.subplot(421), plt.scatter(t_ir,series[:,0],s=5,c='k')
plt.subplot(423), plt.scatter(t_ir,series[:,1],s=5,c='k')
plt.subplot(425), plt.scatter(t_ir,series[:,2],s=5,c='k')
plt.subplot(427), plt.scatter(t_ir,series[:,3],s=5,c='k')
plt.xlabel("Time (day)")
plt.ylabel("Radial velocity (m/s)²")
plt.subplot(422), plt.plot(freq, LS_p[:,0] )
plt.subplot(424), plt.plot(freq, LS_p[:,1] )
plt.subplot(426), plt.plot(freq, LS_p[:,2] )
plt.subplot(428), plt.plot(freq, LS_p[:,3] )
plt.xlabel("Frequency (day⁻¹)")
plt.ylabel("Power")

#%%
#Test with star with 4 planets
y = time_series[:,4]
f1,f2,f3,f4 = freq_pl[1],freq_pl[2],freq_pl[3],freq_pl[4]

series, LS_p = generate_4_periodogrammes(t_ir,freq,y)

plt.figure(4)
plt.suptitle("Time series with their Lomb-Scargle periodogramme generated by removing the best frequency fit\n 4 planets")
plt.subplot(421), plt.scatter(t_ir,series[:,0],s=5,c='k')
plt.subplot(423), plt.scatter(t_ir,series[:,1],s=5,c='k')
plt.subplot(425), plt.scatter(t_ir,series[:,2],s=5,c='k')
plt.subplot(427), plt.scatter(t_ir,series[:,3],s=5,c='k')
plt.xlabel("Time (day)")
plt.ylabel("Radial velocity (m/s)²")

plt.subplot(422)
plt.plot(freq, LS_p[:,0] )
plt.plot([f1,f1] , [min(LS_p[:,0]),max(LS_p[:,0])],'r--')
plt.plot([f2,f2] , [min(LS_p[:,0]),max(LS_p[:,0])],'g--')
plt.plot([f3,f3] , [min(LS_p[:,0]),max(LS_p[:,0])],'b--')
plt.plot([f4,f4] , [min(LS_p[:,0]),max(LS_p[:,0])],'y--')

plt.subplot(424), plt.plot(freq, LS_p[:,1] )
plt.plot([f1,f1] , [min(LS_p[:,1]),max(LS_p[:,1])],'r--')
plt.plot([f2,f2] , [min(LS_p[:,1]),max(LS_p[:,1])],'g--')
plt.plot([f3,f3] , [min(LS_p[:,1]),max(LS_p[:,1])],'b--')
plt.plot([f4,f4] , [min(LS_p[:,1]),max(LS_p[:,1])],'y--')

plt.subplot(426), plt.plot(freq, LS_p[:,2] )
plt.plot([f1,f1] , [min(LS_p[:,2]),max(LS_p[:,2])],'r--')
plt.plot([f2,f2] , [min(LS_p[:,2]),max(LS_p[:,2])],'g--')
plt.plot([f3,f3] , [min(LS_p[:,2]),max(LS_p[:,2])],'b--')
plt.plot([f4,f4] , [min(LS_p[:,2]),max(LS_p[:,2])],'y--')

plt.subplot(428), plt.plot(freq, LS_p[:,3] )
plt.plot([f1,f1] , [min(LS_p[:,3]),max(LS_p[:,3])],'r--')
plt.plot([f2,f2] , [min(LS_p[:,3]),max(LS_p[:,3])],'g--')
plt.plot([f3,f3] , [min(LS_p[:,3]),max(LS_p[:,3])],'b--')
plt.plot([f4,f4] , [min(LS_p[:,3]),max(LS_p[:,3])],'y--')
plt.xlabel("Frequency (day⁻¹)")
plt.ylabel("Power")