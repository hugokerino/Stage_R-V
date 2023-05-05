#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 14:51:43 2023

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
    # activity + granulation + the intrinsic errors + planet if params_pl
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
dt = Ttot*facteur_surech/(N-1)

# Irregular sampling 1 : with normal distribution
facteur_ech = 10
t_ir1 = np.zeros(Ttot)

ind0 = round( 0 + facteur_ech*np.abs(np.random.normal(0, 0.1)))
t_ir1[0] = t[ ind0 ]
ind_fin = round( (N-1) - facteur_ech*np.abs(np.random.normal(0, 0.1)))
t_ir1[-1] = t[ ind_fin ]

pas = N/Ttot
for i in range(Ttot-2):
    ind = round((i+1)*pas+facteur_ech*np.random.normal(0, 0.1))
    t_ir1[i+1] =  t[ind]
    
# Irregular sampling 2 :Random sampling
t_ir2 = np.zeros(Ttot)
indices = np.sort(np.random.choice(N,Ttot,replace=False))
count=0
for i in range(Ttot):
    t_ir2[i]  =  t[indices[count]]
    count +=1

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

amp = 3
epsilon = 1
tau = 150
# amp  = gamma.rvs(2.0, 0.5) 
# epsilon = uniform.rvs(0.5, 1)
# tau = np.random.normal(3*Prot, 0.1*Prot) 
gam  = 2.0/epsilon #not change
logP = np.log(Prot) #not change 
met  = 1.0 # not change 

params_act = [amp, gam, logP, met]

##Planet
Ppl = 50
K=5
T0 = Ppl/2
# Ppl = random.uniform(10*dt,Ttot/2)
# K = loguniform.rvs(0.1, 10)
# T0 =  random.uniform(0,Ppl)

params_pl = [Ppl,K,T0]
#params_pl = 0


#%%
#Create the time series
y_noise, yerr = generate_regular_data_H0(params_gr, params_act, N, t)
y_noise1, yerr_ir1 = generate_regular_data_H0(params_gr, params_act, Ttot, t_ir1)
y_noise2, yerr_ir2 = generate_regular_data_H0(params_gr, params_act, Ttot, t_ir2)

#Add planete signal if there is one
if (params_pl !=0):
    ypl = generate_planete(params_pl, t)
    ypl1= generate_planete(params_pl,t_ir1)
    ypl2= generate_planete(params_pl,t_ir2)
    y_ir1 = ypl1 + y_noise1
    y_ir2 = ypl2 + y_noise2
    y = ypl + y_noise
    
else:
    y_ir1 = ypl1 + y_noise1
    y_ir2 = ypl2 + y_noise2
    y = ypl + y_noise1

#%%
#Plot
plt.close("all")
sigma1 = np.std(yerr_ir1)
sigma2 = np.std(yerr_ir2)

plt.figure(num=1,figsize=(12,6))

plt.subplot(511) 
plt.plot(t,y),plt.title("With oversampling")

plt.subplot(513) 
plt.errorbar(t_ir1,y_ir1,zorder=0), plt.errorbar(t_ir1,y_ir1, sigma1,fmt='k.', zorder = 1), plt.title("Iregular sampling 1 : quasi-uniformly distributed")

plt.subplot(515) 
plt.errorbar(t_ir2,y_ir2,zorder=0), plt.errorbar(t_ir2,y_ir2, sigma2,fmt='k.', zorder = 1), plt.title("Iregular sampling 2 : randomly choice")
    
#%%
#Frequency analysis
fmin = 1/Ttot
fmax = (1/dt)/2
freq = np.arange(fmin,fmax,fmin/10)

#Irregular sampling 1
start = time.time()
GLS_ir1 = LombScargle(t_ir1, y_ir1,sigma1*np.ones(len(y_ir1)),normalization='standard').power(freq,method='cython')
diff2 = time.time() - start

#Irregular sampling 2
GLS_ir2 = LombScargle(t_ir2, y_ir2,sigma2*np.ones(len(y_ir2)),normalization='standard').power(freq,method='cython')

#plot
plt.figure(2)
plt.suptitle("Lomb-Scargle periodograme")
plt.subplot(121)
plt.plot(freq,GLS_ir1,label='Quasi uniformly time sampling')
plt.plot(freq,GLS_ir2,label='Randomly sampling')

if(params_pl != 0):    
    plt.plot([(1/Ppl),(1/Ppl)],[min(min(GLS_ir1),min(GLS_ir2)), max(max(GLS_ir1),max(GLS_ir2))],'r--')

plt.xlabel("Frequence [d⁻¹]")
plt.ylabel("Amplitude(m/s)²")
plt.legend()

plt.subplot(122)
plt.semilogx(np.flip(1/freq),np.flip(GLS_ir1), label='Quasi uniformly time sampling' )
plt.semilogx(np.flip(1/freq),np.flip(GLS_ir2),label='Randomly sampling')

if(params_pl != 0):    
    plt.plot([(Ppl),(Ppl)],[min(min(GLS_ir1),min(GLS_ir2)), max(max(GLS_ir1),max(GLS_ir2))],'r--')

plt.xlabel("Time (days)")
plt.ylabel("Amplitude(m/s)²")
plt.legend()
