import numpy as np
import matplotlib.pyplot as plt

delta_t = 0.00005
ts = np.linspace(0,25,int(25/delta_t))

delta = 0
omega = 1
hbar = 1
epsilon0 = 1
V = 1
mu21 = 1
g = -1j*(omega/(2*hbar*epsilon0*V))**(1/2)*mu21

c11i = 1/np.sqrt(2)
c20i = 0
c12i = 1/np.sqrt(2)
c21i = 0

c11 = [c11i]
c20 = [c20i]
c12 = [c12i]
c21 = [c21i]

ns = [1,2]
c1n = [c11, c12]
c2nm1 = [c20, c21]

for i in range(len(ns)):
    n = ns[i]
    gn = np.sqrt(n)*g
    Omegan = np.sqrt(delta**2 + 4*(np.abs(gn)**2))
    for t in ts.tolist()[1:]:
        cosTerm = np.cos(Omegan*t/2)
        sinTerm = np.sin(Omegan*t/2)
        coeff_c = -2j*np.conjugate(gn)/Omegan
        coeff = -2j*gn/Omegan
        c1n[i].append(cosTerm*c1n[i][0] + coeff_c*sinTerm*c2nm1[i][0])
        c2nm1[i].append(coeff*sinTerm*c1n[i][0] + cosTerm*c2nm1[i][0])


c11_sqamp = np.real(np.array([a*np.conjugate(a) for a in c1n[0]]))
c12_sqamp = np.real(np.array([a*np.conjugate(a) for a in c1n[1]]))
c20_sqamp = np.real(np.array([a*np.conjugate(a) for a in c2nm1[0]]))
c21_sqamp = np.real(np.array([a*np.conjugate(a) for a in c2nm1[1]]))


# general plots
plt.plot(ts, c11_sqamp, color='r', linestyle='-', alpha=0.15,
         label=r'|1,1$\rangle$')
plt.plot(ts, c12_sqamp, color='r', linestyle='--', alpha=0.15,
         label=r'|1,2$\rangle$')
plt.plot(ts, c20_sqamp, color='b', linestyle='-', alpha=0.15,
         label=r'|2,0$\rangle$')
plt.plot(ts, c21_sqamp, color='b', linestyle='--', alpha=0.15,
         label=r'|2,1$\rangle$')
plt.plot(ts, c21_sqamp+c20_sqamp,
         color='black', linestyle='-', label=r'excited state')
plt.ylabel(r'amplitude', fontsize='xx-large')
plt.xlabel("t",fontsize='xx-large')
plt.legend(fontsize='xx-large')
plt.grid()
plt.tight_layout()
plt.show()
