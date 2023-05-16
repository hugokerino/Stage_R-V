#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 10:36:31 2023

@author: hkerino
"""
import numpy as np
from matplotlib import pyplot as plt
from astropy.timeseries import LombScargle
from scipy.integrate import simps
import time


def lorentzian_components(params_gr, t_d):
    
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
    t = t_d*24*3600
    N    = len(t)
    Ttot = (max(t)-min(t)) # total time in seconds
    dt = np.median(np.diff(t)) # sampling rate
    freq_Nyq = (1.0/dt)/2 # Nyquist frequency
    freq  = np.arange(1.0/Ttot,freq_Nyq,1.0/Ttot, dtype='float64')
    # freq  = np.linspace(1.0/Ttot, 1./2/dt, len(t)) # Hz

    # define the power spectra as a sum of 3 components for granulation, mesogranulation and supergranulation + oscillations
    A1, A2, A3, B1, B2, B3, C1, C2, C3, AL, Gm, nu0, cste = params_gr 
    
    VPSD = (A1 / (1+(B1*freq)**C1) + A2 / (1+(B2*freq)**C2) +  A3 / (1+(B3*freq)**C3) + \
                AL * (Gm**2/((freq-nu0)**2+Gm**2)) +  cste  ) 
    
    #VPSD /= 1e6     # units of VPSD is (m/s)**2/Hz
    VPSD *= (freq[-1]-freq[0])*10e6
    # take random phase between 0 and 2pi
    
    phase = 2*np.pi*np.random.rand(1,len(VPSD))
    
    # Synthetic radial velocity measurements
    ysyn = np.zeros(N)
    for i in range(N):
        ysyn[i] = np.sum(np.sqrt(VPSD)*np.sin(2*np.pi*t[i]*freq+phase))
       
    return freq, VPSD, ysyn



# 1. Define parameters for the granulation+oscillation signals
# Example here alphaCenA (a solar-like star) from Table 2 of Dumusque et al. 2011
A1, A2, A3  = 0.027, 0.003, 0.3*1e-3 # m/s
B1, B2, B3  = 7.4*3600, 1.2*3600, 17.9*60 # seconds
C1, C2, C3  = 3.1, 3.9, 8.9 # dimensionless
AL, Gm, nu0 = 2.6*1e-3, 0.36*1e-3, 2.4*1e-3 # (m/s), Hz, Hz
cste        = 1.4e-4 # (m/s)**2/Hz

params_gr = [A1, A2, A3, B1, B2, B3, C1, C2, C3, AL, Gm, nu0, cste]


N = 201 # number of data point in a regularly sampled grid
Ttot = 200 # days
t = np.linspace(0,Ttot,N)
dt = Ttot/(N-1)

start = time.time()
freq, DSP_cible, y_real = lorentzian_components(params_gr, t)
diff = time.time() - start
print(diff)

#Gneration of many time series and their periodogramme
Nbr_test = 1000
P = []
P_GLS = []
Y = []

for i in range(Nbr_test):
    y = lorentzian_components(params_gr, t)
    Y.append(y[2])
    p_i = 1.0/len(y[2]) * np.abs(np.fft.fftshift(np.fft.fft(y[2])))**2 
    P.append(p_i[round(len(p_i)/2)+1:len(p_i)-1])
    P_GLS.append(LombScargle(t*3600*24, y[2],normalization='psd').power(y[0],method='cython'))

    

#%%
Y_moy = np.mean(Y, axis = 0)
P_moy = np.mean(P, axis = 0)
P_GLS_moy = np.mean(P_GLS, axis = 0)

freq_p = np.linspace(freq[0],freq[-1],len(P_moy))
freq_g = np.linspace(freq[0],freq[-1],len(P_GLS_moy))

#%%
plt.close("all")
plt.figure(1)
plt.subplot(511), plt.plot(freq, DSP_cible),plt.title("DSP cible") #
plt.subplot(513), plt.plot(freq_p, P_moy),plt.title("Periodogramme moyenné")
plt.subplot(515), plt.plot(freq_g, P_GLS_moy ), plt.title("GLS moyenné")


plt.figure(2)
plt.plot(freq, DSP_cible*(len(freq)/2)+2, label='DSP theorique')
plt.plot(freq_p, P_moy,     label='Periodogramme moyenne')  
plt.plot(freq_g, P_GLS_moy, label = 'GLS moyenne')
plt.xlabel("frequence (Hz)")
plt.ylabel("Amplitude (m/s)**2")
plt.legend()


plt.figure(3)
plt.plot(freq, P[0], label='ex DSP')