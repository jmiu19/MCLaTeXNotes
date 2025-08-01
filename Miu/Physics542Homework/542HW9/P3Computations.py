import numpy as np
import random
import matplotlib.pyplot as plt
import math

def Theta(x):
    if x<0:
        return 0
    else:
        return 1

def I(t, Ns):
    I = 0
    for n in Ns:
        I += Theta(t+0.1-n) - Theta(t-0.1-n)
    return I

def prodI_withTau(t,tau,Ns):
    prod = 0
    return I(tau+t,Ns)*I(t,Ns)

Ns = np.linspace(-50,50, 101).tolist()
ts = np.linspace(-5,5, 1001).tolist()

I_ts = [0]*len(ts)
for i in range(len(ts)):
    t = ts[i]
    I_ts[i] = I(t, Ns)


Ibar = np.sum(I_ts)/len(ts)

g2_taus = []
taus = np.linspace(-2,2, 401).tolist()
for i in range(len(taus)):
    tau = taus[i]
    prod_tau_tsSum = 0
    for t in ts:
        prod_tau_tsSum += prodI_withTau(t,tau,Ns)
    prod_tau = prod_tau_tsSum/len(ts)
    g2_taus.append(prod_tau/Ibar)

g2_taus_ana = []
for i in range(len(taus)):
    tau = abs(taus[i])
    if abs(tau-round(tau))>0.2:
        g2 = 0
    else:
        g2 = 1 - 5*(abs(tau-round(tau)))
    g2_taus_ana.append(g2)


plt.plot(taus, I_ts[299:700], label='I')
plt.plot(taus, g2_taus, label=r'$g^{(2)}$')
plt.plot(taus, g2_taus_ana, label=r'Eq. (*)')
plt.axvline(x=0.2, color='black', linestyle='--', alpha=0.2)
plt.axvline(x=0.8, color='black', linestyle='--', alpha=0.2)
plt.xlabel(r"t",fontsize='x-large')
plt.legend()
plt.show()
