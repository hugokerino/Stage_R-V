#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Generate synthetic dataset based on a QP kernel

Created on Fri Apr 28 15:46:40 2023

@author: ssulis
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# to be installed --> https://george.readthedocs.io/en/latest/user/quickstart/
import george
from george import kernels

import random 
from scipy.stats import loguniform, gamma, uniform

import warnings
warnings.filterwarnings("ignore")


#%%***************************************************************************#
# Parameters
#*****************************************************************************#
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


def generate_regular_data(params_gr, params_act, params_pl, N, t):
    
    '''
    Inputs:
    - params_gr:= 
            see function lorentzian_components(params_gr, t)
    - params_act:=
            - amplitude (GP amplitude)
            - gam = 2/epsilon in the paper (GP gamma)
            - logP = log(P) in the paper (GP period)
            - met = 1.0 in the paper (GP metric)
    - params_pl
            - Ppl (Orbital period)
            - K (semi-amplitude)
            - T0 ( Phase à l'origine)
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
    # activity + granulation + the intrinsic errors + planet if params_pl
    ytot = ygr + yact + yerr * np.random.randn(N)
    if ( params_pl != 0):
        Ppl, K, T0 = params_pl
        ypl = K*np.sin(2*np.pi*(t-T0)/Ppl)
        ytot += ypl 
        return ytot, yerr
    
    return ytot, yerr


#%% ===========================================================================
# Choose the input parameters =================================================
# =============================================================================

N = 200 # number of data point in a regularly sampled grid
Ttot = 200 # days
t    = np.linspace(0,Ttot,N) #days


# 1. Define parameters for the granulation+oscillation signals
# Example here alphaCenA (a solar-like star) from Table 2 of Dumusque et al. 2011
A1, A2, A3  = 0.027, 0.003, 0.3*1e-3 # m/s
B1, B2, B3  = 7.4*3600, 1.2*3600, 17.9*60 # seconds
C1, C2, C3  = 3.1, 3.9, 8.9 # dimensionless
AL, Gm, nu0 = 2.6*1e-3, 0.36*1e-3, 2.4*1e-3 # (m/s), Hz, Hz
cste        = 1.4e-4 # (m/s)**2/Hz

params_gr = [A1, A2, A3, B1, B2, B3, C1, C2, C3, AL, Gm, nu0, cste]

# 2. Define parameters for the activity signal
Prot = 62 # days -- Prot to be chosen randomly in HARPS sample (see paper)
logP = np.log(Prot) # to not change this line

amp = 3
epsilon = 1
tau = 150
#amp  = gamma.rvs(2.0, 0.5) 
#epsilon = uniform.rvs(0.5, 1)
#tau = np.random.normal(3*Prot, 0.1*Prot) 

gam  = 2.0/epsilon # to not change this line
met  = 1.0 # to not change this line

params_act = [amp, gam, logP, met]

# 3. Define parmeters for the planet
delta_t = Ttot/(N-1)
Ppl = 50
K=5
T0 = Ppl/2
# Ppl = random.uniform(10*delta_t,Ttot/2)
# K = loguniform.rvs(0.1, 10)
# T0 =  random.uniform(0,Ppl)

params_pl = [Ppl,K,T0]
#params_pl = 0

#%% ===========================================================================
# Generate the synthetic time series under H0 with a regular sampling 
# =============================================================================

y, yerr = generate_regular_data(params_gr, params_act, params_pl, N, t)


#%% ===========================================================================
# check the result
# =============================================================================


# test: calcul classical periodogram
dt = np.min(t[1:]-t[:-1])*24*3600
f  = np.linspace(-1./2/dt, 1./2/dt, len(t)) 
p = 1.0/len(y) * np.abs(np.fft.fftshift(np.fft.fft(y)))**2

yerr*=0
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.errorbar(t,y, yerr, zorder=0)
plt.errorbar(t,y, yerr,fmt='k.', zorder=1)
plt.xlabel('Time (days)')
plt.ylabel('RV (m/s)')
plt.title("Example of simulated data");
plt.subplot(122)
# plt.loglog(f*1e3,p,'k')
plt.semilogx((1.0/f)/24/3600,p,'k')
plt.plot([Prot, Prot],[min(p), max(p)],'r--')
# plt.xlabel('Frequency (mHz)')
plt.xlabel('Periods (days)')
plt.ylabel('Pcl (m2/s2)')
plt.title("Check associated periodogram");


