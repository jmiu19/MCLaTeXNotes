import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import special
import scipy.optimize as opt
from scipy.optimize import curve_fit
import os
import sys

# Read in data here
outerRing = pd.read_csv('OuterRing.csv',header=0)
innerRing = pd.read_csv('InnerRing.csv',header=0)


## space for doing calculations
def linearFit(x, m):
  """
  input:
    x: array, independent data
    m: slope
  output:
    y=xm
  """
  return x*m

OsinThe = np.sin(outerRing['theta_1'].values)
IsinThe = np.sin(innerRing['theta_1'].values)
V = outerRing['HV'].values*1000


Oparams, Oparams_covariance = opt.curve_fit(linearFit, V**(-1/2), OsinThe, p0 = [1])
Operr = np.sqrt(np.diag(Oparams_covariance))  ## np.diag returns the diagonal elements [0,0] and [1,1]
                                              ## the standard deviation is the square root of the variance

Iparams, Iparams_covariance = opt.curve_fit(linearFit, V**(-1/2), IsinThe, p0 = [1])
Iperr = np.sqrt(np.diag(Oparams_covariance))  ## np.diag returns the diagonal elements [0,0] and [1,1]
                                              ## the standard deviation is the square root of the variance

h = (6.626*10**(-34))
e = (1.602*10**(-19))
m = (9.109*10**(-31))

Od = h/(2*Oparams[0]*np.sqrt(2*e*m))
Id = h/(2*Iparams[0]*np.sqrt(2*e*m))

Od_nm = Od*10**9
Id_nm = Id*10**9

Idsig = Iperr[0]*Id/Iparams[0]
Odsig = Operr[0]*Od/Oparams[0]

Idsig_nm = Idsig*10**9
Odsig_nm = Odsig*10**9

## It is always good to first plot data you either generate or read in
plt.scatter(V**(-1/2), OsinThe, c='blue')
plt.grid()
plt.title('Outer Ring', fontsize='xx-large')
plt.ylabel(r'$\sin(\theta)$', fontsize='xx-large')
plt.xlabel('$V^{-1/2}$', fontsize='xx-large')
plt.show()

plt.scatter(V**(-1/2), IsinThe, c='blue')
plt.grid()
plt.title('Inner Ring', fontsize='xx-large')
plt.ylabel(r'$\sin(\theta)$', fontsize='xx-large')
plt.xlabel('$V^{-1/2}$', fontsize='xx-large')
plt.show()





plt.scatter(V**(-1/2), OsinThe, c='blue')
plt.plot(V**(-1/2), linearFit(V**(-1/2), Oparams[0]), color='red')
plt.grid()
plt.title('Outer Ring', fontsize='xx-large')
plt.ylabel(r'$\sin(\theta)$', fontsize='xx-large')
plt.xlabel('$V^{-1/2}$', fontsize='xx-large')
plt.annotate('slope = %.4f '%Oparams[0], (0.53,0.35),
               xycoords='figure fraction',
               fontsize='x-large')
plt.annotate('d$_{estimated}$ = %.4f nm'%Od_nm, (0.53,0.29),
               xycoords='figure fraction',
               fontsize='x-large')
plt.annotate('st. dev. = %.4f nm'%Odsig_nm, (0.53,0.23),
               xycoords='figure fraction',
               fontsize='x-large')
plt.show()

plt.scatter(V**(-1/2), IsinThe, c='blue')
plt.plot(V**(-1/2), linearFit(V**(-1/2), Iparams[0]), color='red')
plt.grid()
plt.title('Inner Ring', fontsize='xx-large')
plt.ylabel(r'$\sin(\theta)$', fontsize='xx-large')
plt.xlabel('$V^{-1/2}$', fontsize='xx-large')
plt.annotate('slope = %.4f '%Iparams[0], (0.53,0.35),
               xycoords='figure fraction',
               fontsize='x-large')
plt.annotate('d$_{estimated}$ = %.4f nm'%Id_nm, (0.53,0.29),
               xycoords='figure fraction',
               fontsize='x-large')
plt.annotate('st. dev. = %.4f nm'%Idsig_nm, (0.53,0.23),
               xycoords='figure fraction',
               fontsize='x-large')
plt.show()



d1 = 0.142*np.sin(np.pi/3)*10**(-9)
d2 = 0.142*(np.sin(np.pi/6)+1)*10**(-9)

print('expected', 'data', 'sig')
print(d1, Od, Odsig)
print(d2, Id, Idsig)

print(np.abs(d1-Od)/Odsig)
print(np.abs(d2-Id)/Idsig)
