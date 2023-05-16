#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 13:52:15 2023

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
import math

#%%
#functions 

def LSfitw(y,M,Sigma_inv):
    #finds beta such that ||y-M*beta||*2 is minimal but with diagonal weight
    #Sigma_inv should be Sigma^^(-1/2) so iv std matrix
    Mt = np.dot(Sigma_inv,M)
    yt = np.dot(Sigma_inv,y)
    A =np.dot(Mt.T,Mt) # this and the following steps implement the LS iversion formula
    B = np.dot(Mt.T,yt)
    betaest = np.dot(np.linalg.inv(A),B)
    yest = np.dot(M,betaest)
    return yest


def my_GLS(t,y,EB, periods):
    Sigma_inv = np.diagflat(1./EB)
    power=[]
    N = len(t)
    un = np.ones(np.shape(y)) # a vector of ones
    unt = np.dot(Sigma_inv,un)
    yt = np.dot(Sigma_inv,y)
    yest0 = (np.dot(yt,unt))/(np.dot(unt,unt))*un
    RSS0 = np.sum(((y-yest0)/EB)**2)
    for i in range(len(periods)):
        # create M0 = constant
        # create M1 = constant + sinusoid
        M1 = (np.concatenate(([un], [np.cos(2*np.pi*t/periods[i])], [np.sin(2*np.pi*t/periods[i])]), axis=0)).T # the design matric M : shape is 20 *2        
        yest1 = LSfitw(y,M1,Sigma_inv)
        RSS1 = np.sum(((y-yest1)/EB)**2)
        power.append((RSS0-RSS1)/RSS0)
    return power

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
    VPSD *= (freq[-1]-freq[0])*1e6     # units of VPSD is (m/s)**2/Hz

           
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
    yact = gp.sample(ts)
    
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



def print_4periodogramme(LS_p,GLS_p,label,freq_pl,i):
    LS = LS_p[:,i,:]
    GLS = GLS_p[:,i,:]
    f1,f2,f3,f4 = freq_pl[i,:]
    l1,l2,l3,l4 = label[i,:]
    
    plt.figure()
    plt.suptitle("4 Lomb-Scargle peridogramme generated by removing the best fit\n"+str(i)+" planet")
    plt.subplot(411),plt.plot(freq, LS[:,0] , label =l1), plt.legend()
    plt.scatter(freq, GLS[:,0], s=5, c='k')
    if (f1!=0) : plt.plot([f1,f1] , [min(LS[:,0]),max(LS[:,0])],'r--')
    if (f2!=0) : plt.plot([f2,f2] , [min(LS[:,0]),max(LS[:,0])],'g--')
    if (f3!=0) : plt.plot([f3,f3] , [min(LS[:,0]),max(LS[:,0])],'b--')
    if (f4!=0) : plt.plot([f4,f4] , [min(LS[:,0]),max(LS[:,0])],'y--')

    plt.subplot(412), plt.plot(freq, LS[:,1] , label =l2), plt.legend()
    plt.scatter(freq, GLS[:,1], s=5, c='k')
    if (f1!=0) : plt.plot([f1,f1] , [min(LS[:,1]),max(LS[:,1])],'r--')
    if (f2!=0) : plt.plot([f2,f2] , [min(LS[:,1]),max(LS[:,1])],'g--')
    if (f3!=0) : plt.plot([f3,f3] , [min(LS[:,1]),max(LS[:,1])],'b--')
    if (f4!=0) : plt.plot([f4,f4] , [min(LS[:,1]),max(LS[:,1])],'y--')

    plt.subplot(413), plt.plot(freq, LS[:,2] , label =l3), plt.legend()
    plt.scatter(freq, GLS[:,2], s=5, c='k')
    if (f1!=0) : plt.plot([f1,f1] , [min(LS[:,2]),max(LS[:,2])],'r--')
    if (f2!=0) : plt.plot([f2,f2] , [min(LS[:,2]),max(LS[:,2])],'g--')
    if (f3!=0) : plt.plot([f3,f3] , [min(LS[:,2]),max(LS[:,2])],'b--')
    if (f4!=0) : plt.plot([f4,f4] , [min(LS[:,2]),max(LS[:,2])],'y--')

    plt.subplot(414 ), plt.plot(freq, LS[:,3] , label = l4), plt.legend()
    plt.scatter(freq, GLS[:,3], s=5, c='k')
    if (f1!=0) : plt.plot([f1,f1] , [min(LS[:,3]),max(LS[:,3])],'r--')
    if (f2!=0) : plt.plot([f2,f2] , [min(LS[:,3]),max(LS[:,3])],'g--')
    if (f3!=0) : plt.plot([f3,f3] , [min(LS[:,3]),max(LS[:,3])],'b--')
    if (f4!=0) : plt.plot([f4,f4] , [min(LS[:,3]),max(LS[:,3])],'y--')
    
    plt.xlabel("Frequency (day⁻¹)")
    plt.ylabel("Power")
    

def generate_4_periodogrammes(t,f, y, freq_pl,nbr_planete):
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
    y_t = np.zeros((len(t), 4)) #time series
    LS  = np.zeros((len(f), 4)) # 4 lombscargle 
    GLS  = np.zeros((len(f), 4)) # 4 lombscargle 
    nbr_planete_now = np.copy(nbr_planete)
    label= np.zeros(4) #(1 if max = freq_pl, 0 else)
    max_value = np.zeros((2,4)) # (argmax, max)
    
    y_t[:,0] = y 
    
    for i in range(3):
        LS_i =  LombScargle(t, y_t[:,i])
        LS[:,i] = LS_i.power(f,method='cython')
        best_freq = round(f[np.argmax(LS[:,i])],3)
        #print("meilleur fit : ",best_freq)
        max_value[0,i], max_value[1,i] = best_freq, np.max(LS[:,i])
        
        GLS[:,i] = my_GLS(t,y_t[:,i], np.ones(len(t)) ,1/f ) 
                
        if ( any(np.isclose(freq_pl, best_freq, atol=0.002)) ):
            label[i] = 1
            freq_pl[ np.where( np.isclose(freq_pl, best_freq, atol=0.002) ) ] = 0
            nbr_planete_now[i+1:] = nbr_planete_now[i] - 1
        else:
            label[i] = 0
            
        y_fit = LS_i.model(t,best_freq)
        y_t[:,i+1] = y_t[:,i] - y_fit
      
    LS_i = LombScargle(t, y_t[:,3])
    LS[:,3] = LS_i.power(f,method='cython')
    best_freq = round(f[np.argmax(LS[:,3])],3)
    #print("meilleur fit : ",best_freq)
    max_value [0,3], max_value[1,3] =  best_freq, np.max(LS[:,3])
    
    GLS[:,3] = my_GLS(t,y_t[:,3], np.ones(len(t)) ,1/f ) 
    
    if (any(np.isclose(freq_pl, best_freq, atol=0.002))):
        label[3] = 1
        freq_pl[np.where( np.isclose(freq_pl, best_freq, atol=0.002) )] = 0
    else:
        label[3] = 0
    
    #print("label = ", label)
    
    return LS, label, max_value, nbr_planete_now, GLS



def generate_data(t,dt,Ttot,freq,params_gr, params_act):
    
    time_series = np.zeros((len(t),5))
    LS = np.zeros((len(freq),5,4))
    GLS = np.zeros((len(freq),5,4))
    freq_pl = np.zeros((5,4))
    nbr_planete = np.zeros((5,4))
    label = np.zeros((5,4))
    max_value = np.zeros((2,5,4))
    
    y_noise, yerr = generate_regular_data_H0(params_gr, params_act, len(t), t) 
    
    #print("freq planet : ",freq_pl[0,:])
    time_series[:,0] = y_noise #Without planet
    LS[:,0,:], label[0,:], max_value[:,0,:],nbr_planete[0,:], GLS[:,0,:] = generate_4_periodogrammes(t, freq, np.copy(y_noise), freq_pl[0,:],nbr_planete[0,:]) #Without planet
    
    for i in range(1,5):
        y_i = y_noise
        for k in range(1,i+1):
            Ppl =  random.uniform(10*dt,Ttot/2) 
            K = loguniform.rvs(0.1, 10)
            T0 =  random.uniform(0,Ppl)
            params_pl = [Ppl,K,T0]
            freq_pl[i,k-1] = round(1/Ppl,3)
            
            y_pl = generate_planete(params_pl, t)
            y_i += y_pl 
        
        
        nbr_planete[i,:] = i
        #print("freq planet : ",freq_pl[i,:])
        time_series[:,i] = y_i
        LS[:,i,:], label[i,:], max_value[:,i,:],nbr_planete[i,:], GLS[:,i,:] = generate_4_periodogrammes(t, freq, y_i,np.copy(freq_pl[i,:]),nbr_planete[i,:])
        
        
    return time_series, LS, label, max_value, freq_pl,nbr_planete,GLS

#%%
#Parameters for activity signal
# Grannulation + Oscillation
A1, A2, A3  = 0.027, 0.003, 0.3*1e-3 # m/s
B1, B2, B3  = 7.4*3600, 1.2*3600, 17.9*60 # seconds
C1, C2, C3  = 3.1, 3.9, 8.9 # dimensionless
AL, Gm, nu0 = 2.6*1e-3, 0.36*1e-3, 2.4*1e-3 # (m/s), Hz, Hz
cste        = 1.4e-4 # (m/s)**2/Hz

params_gr = [A1, A2, A3, B1, B2, B3, C1, C2, C3, AL, Gm, nu0, cste]

##Activity signal
Prot = random.randint(10,90)*24*3600#62 # days -- Prot to be chosen randomly in HARPS sample (see paper)

amp  = gamma.rvs(2.0, 0.5) 
epsilon = uniform.rvs(0.5, 1)
tau = np.random.normal(3*Prot, 0.1*Prot) 
gam  = 2.0/epsilon #not change
logP = np.log(Prot) #not change 
met  = 1.0 # not change 

params_act = [amp, gam, logP, met]

#%%
#Over-sampling
Ttot = 200 # days
facteur_surech = 10
N = Ttot*facteur_surech +1 # number of data point in a regularly sampled grid
t_over = np.linspace(0,Ttot,N) #days
dt = Ttot*facteur_surech/(N-1) #After sampling


# Irregular sampling 1 : with normal distribution
facteur_ech = 20
t = np.zeros(Ttot)

ind0 = round( 0 + facteur_ech*np.abs(np.random.normal(0, 0.1)))
t[0] = t_over[ ind0 ]
ind_fin = round( (N-1) - facteur_ech*np.abs(np.random.normal(0, 0.1)))
t[-1] = t_over[ ind_fin ]

pas = N/Ttot
for i in range(Ttot-2):
    ind = round((i+1)*pas+facteur_ech*np.random.normal(0, 0.1))
    t[i+1] =  t_over[ind]
    
#irregular sampling 2
# t = np.zeros(Ttot)
# indices = np.sort(np.random.choice(N,Ttot,replace=False))
# count=0
# for i in range(Ttot):
#     t[i]  =  t_over[indices[count]]
#     count +=1

fmin = 1/(t[-1] - t[0])
fmax = (1/dt)/2 
#freq = np.arange(fmin,fmax,fmin/10)
freq = np.linspace(fmin,fmax,990)
#%%
#Generate the data

time_series, LS, label, max_value,freq_pl,nbr_planete,GLS = generate_data(t, dt, Ttot, freq, params_gr, params_act)

        
#%%
#plot

plt.close('all')
indice = 4

print_4periodogramme(LS,GLS, label,freq_pl,indice)

# plt.figure()
# plt.title('Original time serie, ' + str(indice) + ' planet')
# plt.scatter(t,time_series[:,indice],s=5,c='k'), plt.plot(t,time_series[:,indice])



#%%
# Set 1
def create_set1(nbr_periodogramme, pourcentage_positive,t, dt, Ttot, freq):
    
    count_positif = 0
    count_negatif = 0
    nbr_ech = nbr_periodogramme/5
    count_ech = np.zeros(5)
    
    X_train = []
    Y_train = []
    
    start = time.time()
    
    A1, A2, A3  = 0.027, 0.003, 0.3*1e-3 # m/s
    B1, B2, B3  = 7.4*3600, 1.2*3600, 17.9*60 # seconds
    C1, C2, C3  = 3.1, 3.9, 8.9 # dimensionless
    AL, Gm, nu0 = 2.6*1e-3, 0.36*1e-3, 2.4*1e-3 # (m/s), Hz, Hz
    cste        = 1.4e-4 # (m/s)**2/Hz

    params_gr = [A1, A2, A3, B1, B2, B3, C1, C2, C3, AL, Gm, nu0, cste]
    
    
    #create 40% positif label
    while( count_positif < pourcentage_positive*nbr_periodogramme) :
        ##Activity signal
        Prot = 62 # days -- Prot to be chosen randomly in HARPS sample (see paper)

        amp  = gamma.rvs(2.0, 0.5) 
        epsilon = uniform.rvs(0.5, 1)
        tau = np.random.normal(3*Prot, 0.1*Prot) 
        gam  = 2.0/epsilon #not change
        logP = np.log(Prot) #not change 
        met  = 1.0 # not change 

        params_act = [amp, gam, logP, met]
        
        #Create time serie
        time_series, LS, label, max_value, freq_pl, nbr_planete = generate_data(t, dt, Ttot, freq, params_gr, params_act)

        for i in range(1,5):
            for j in range(4):
                if (label[i,j] == 1):
                    X_train.append((LS[:,i,j],max_value[:,i,j]))
                    Y_train.append(1)
                    count_positif +=1
                    count_ech[ round(nbr_planete[i,j]) ] +=1
    
        print("Nbr positif label : ",count_positif)
        print("Nbr ech par cas de figure : ", count_ech)
        
    print("\nPositive label termined, nbr count positive label = ", count_positif)    
    print("Nbr ech par cas de figure : ", count_ech,"\n")
    nbr_planet_cas_positif = np.copy(count_ech)
    time.sleep(5)
    
    
    #create negatif label 
    while ( (count_negatif < (1-pourcentage_positive)) or (any(count_ech<nbr_ech)) ):
      
        ##Activity signal
        Prot = 62 # days -- Prot to be chosen randomly in HARPS sample (see paper)

        amp  = gamma.rvs(2.0, 0.5) 
        epsilon = uniform.rvs(0.5, 1)
        tau = np.random.normal(3*Prot, 0.1*Prot) 
        gam  = 2.0/epsilon #not change
        logP = np.log(Prot) #not change 
        met  = 1.0 # not change 

        params_act = [amp, gam, logP, met]
        
        #Create time serie
        time_series, LS, label, max_value, freq_pl, nbr_planete = generate_data(t, dt, Ttot, freq, params_gr, params_act)

        for i in range(5):
            for j in range(4):
                
                if ( (label[i,j] == 0)  and (count_ech[ round(nbr_planete[i,j]) ] < nbr_ech) ):
                        X_train.append((LS[:,i,j],max_value[:,i,j]))
                        Y_train.append(label[i,j])
                        count_ech[ round(nbr_planete[i,j]) ] +=1
                        count_negatif += 1
                        
        print("nbr ech ",count_ech)
        print("count negatif = ",count_negatif)

    
    
    print("\nTemps total = ", round(time.time() - start,2))
    print("Nbr ech par cas de figure : ", count_ech)
    print("Nbr positive label : ",count_positif, ", Nbr negative label : ",count_negatif)
    
    print("\nRepartition label positif sur le nombre de planete : ")
    print(" - 0 planete ", 100*round(nbr_planet_cas_positif[0]/count_positif,3),"%")
    print(" - 1 planete ", 100*round(nbr_planet_cas_positif[1]/count_positif,3),"%")
    print(" - 2 planetes ", 100*round(nbr_planet_cas_positif[2]/count_positif,3),"%")
    print(" - 3 planetes ", 100*round(nbr_planet_cas_positif[3]/count_positif,3),"%")
    print(" - 4 planetes ", 100*round(nbr_planet_cas_positif[4]/count_positif,3),"%")
    
    print("Repartition label negatif sur le nombre de planete : ")
    print(" - 0 planete ", 100*round((count_ech[0]-nbr_planet_cas_positif[0])/count_negatif,3),"%")
    print(" - 1 planete ", 100*round((count_ech[1]-nbr_planet_cas_positif[1])/count_negatif,3),"%")
    print(" - 2 planetes ", 100*round((count_ech[2]-nbr_planet_cas_positif[2])/count_negatif,3),"%")
    print(" - 3 planetes ", 100*round((count_ech[3]-nbr_planet_cas_positif[3])/count_negatif,3),"%")
    print(" - 4 planetes ", 100*round((count_ech[4]-nbr_planet_cas_positif[4])/count_negatif,3),"%")
    
    print("\nRepartion classe positif/negatif : ", 100*round(count_positif/len(X_train),2),"% , ", 100*round(count_negatif/len(X_train),2),"%")
    
    print("\nNbr data : ", len(X_train) )
    
    return X_train, Y_train

#%%
nbr_periodogramme_train = 13700
pourcentage_positive = 0.4
#X_train, Y_train = create_set1(nbr_periodogramme_train,pourcentage_positive,t, dt, Ttot, freq)




