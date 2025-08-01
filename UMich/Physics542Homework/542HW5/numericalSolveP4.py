import numpy as np
import matplotlib.pyplot as plt


alphaRs = [1,5]
alphaIs = [1,5]
ts = np.linspace(0,50,1000)

d=1
Ns = range(150)
lam = 1

def measure_d(alphaR, alphaI, t):
    alpha_mag = (alphaR**2 + alphaI**2)**(1/2)
    coeff = d*np.exp(-alpha_mag**2)*alphaI
    list = [-2*(alpha_mag**(2*n))/(np.math.factorial(n)*np.sqrt(n+1))*np.cos(lam*t*np.sqrt(n+1))*np.sin(lam*t*np.sqrt(n+1)) for n in Ns]
    array = np.array(list)
    sum = np.sum(array)
    return coeff*sum

legends = [['1+1i', '1+5i'],['5+1i', '5+5i']]
for i in range(2):
    for j in range(2):
        plt.plot(ts, [measure_d(alphaRs[i], alphaIs[j], t) for t in ts], label=r'$\alpha=$'+legends[i][j])
plt.ylabel(r'$\left\langle \hat{d} \right\rangle$', fontsize='xx-large')
plt.xlabel("t",fontsize='xx-large')
plt.legend(fontsize='xx-large')
plt.grid()
plt.tight_layout()
plt.show()
