UNIQNAME = "jmiu"
NAME = "Jinyan Miao"
COLLABORATORS = "Chi Han, Garen Dye"

import numpy as np                         
from matplotlib import pyplot as plt       
import pandas as pd
from scipy import special                  
import scipy.optimize as opt               
from scipy.optimize import curve_fit       


# Read in data here
blue_data_405 = pd.read_csv('data/blue_data_405.csv',header=0)
red_data_635 = pd.read_csv('data/red_data_635.csv',header=0)
green_data_532 = pd.read_csv('data/green_data_532.csv',header=0)


print(blue_data_405)


y_data_b1 = blue_data_405['Vanode'].values
y_data_b2 = blue_data_405['Vanode0.3'].values
x_data_b1 = blue_data_405['Vcathode'].values +0.001*y_data_b1
x_data_b2 = blue_data_405['Vcathode'].values +0.001*y_data_b2


y_data_r1 = red_data_635['Vanode']
y_data_r2 = red_data_635['Vanode0.3']
x_data_r1 = red_data_635['Vcathode'].values +0.001*y_data_r1
x_data_r2 = red_data_635['Vcathode'].values +0.001*y_data_r2

y_data_g1 = green_data_532['Vanode']
y_data_g2 = green_data_532['Vanode0.3']
x_data_g1 = green_data_532['Vcathode'].values +0.001*y_data_g1
x_data_g2 = green_data_532['Vcathode'].values +0.001*y_data_g2


name = ['635', '532', '405']

## It is always good to first plot data you either generate or read in 
plt.title('Figure 1 (405 nm light)', fontsize='xx-large')
plt.scatter(x_data_b1, y_data_b1, c='blue', label='full intensity')
plt.scatter(x_data_b2, y_data_b2, color=(0.5,0.5,1), label='filtered')
plt.grid()
plt.xlabel('$V_{cathode}$ (V)', fontsize='xx-large')
plt.ylabel('$V_{anode}$ (mV)', fontsize='xx-large')
plt.legend()
plt.show()


## It is always good to first plot data you either generate or read in 
plt.title('Figure 2 (635 nm light)', fontsize='xx-large')
plt.scatter(x_data_r1, y_data_r1, c='red', label='full intensity')
plt.scatter(x_data_r2, y_data_r2, color=(1,0.5,0.5), label='filtered')
plt.grid()
plt.xlabel('$V_{cathode}$ (V)', fontsize='xx-large')
plt.ylabel('$V_{anode}$ (mV)', fontsize='xx-large')
plt.legend()
plt.show()


## It is always good to first plot data you either generate or read in 
plt.title('Figure 3 (532 nm light)', fontsize='xx-large')
plt.scatter(x_data_g1, y_data_g1, c='green', label='full intensity')
plt.scatter(x_data_g2, y_data_g2, color=(0.5,1,0.5), label='filtered')
plt.grid()
plt.xlabel('$V_{cathode}$ (V)', fontsize='xx-large')
plt.ylabel('$V_{anode}$ (mV)', fontsize='xx-large')
plt.legend()
plt.show()



## define the mathematical function in Eq 9 below using a, b, c, and vs as parameters
## and np.heaviside(x1,x2), where we replace x1 and x2 with appropriate parameters

## reminder: to take a quantity X to the nth power in python, use X**n

def Eq9_model(x, a, b, c, vs):
  """ A model to fit the data which is flat above the retarding potential and increases polynomially below the retarding potential

  Args:
    x (array-like) : independent variable
    a (float) : model parameter
    b (float) : model parameter
    c (float) : model parameter
    vs (float) : model parameter, the stopping voltage
  Returns:
    y (array-like) : model output values for each x
  """
  return (a * np.heaviside(vs-x,1) * (vs-x)**3  + b * np.heaviside(vs-x,1) * (vs-x)**4)/(1 + c * np.heaviside(vs-x,1) * (vs-x)**2)



ii=[1,4,1]
jj=[15,19,18]
name = ['635', '532', '405']
colors = ['r', 'g', 'b']

## "The true retarding potential across the phototube is obtained by adding the
##  magnitudes of the potentials measured at the two BNC output jacks."
x_data_1 = [x_data_r1, x_data_g1, x_data_b1]
x_data_2 = [x_data_r2, x_data_g2, x_data_b2]
y_data_1 = [y_data_r1, y_data_g1, y_data_b1]
y_data_2 = [y_data_r2, y_data_g2, y_data_b2]
v_s2 = []


for k in range(3):
  ## use curve_fit starting with the data at index i and ending at index j
  i=ii[k]
  j=jj[k]

  x2 = x_data_2[k]
  y2 = y_data_2[k]

  params, params_covariance = opt.curve_fit(Eq9_model, x2[i:j], y2[i:j], p0 = [1,1,1,0.4])

  perr = np.sqrt(np.diag(params_covariance))  ## np.diag returns the diagonal elements [0,0] and [1,1]
                                              ## the standard deviation is the square root of the variance

  ## plot the data with the fitted function 
  plt.scatter(x2, y2, label='data points', color=colors[k])  ## plot the data
  ## plot the fit line, also starting at index i and ending at index j
  plt.plot(x2[i:j], Eq9_model(x2[i:j], a=params[0], b=params[1], c=params[2], vs=params[3]), 'black', label='Fit')
  plt.plot(np.full(shape=len(x2), fill_value=params[-1].round(4)), y2, color='gray', label='$V_s$ = %.4f'%params[-1].round(4))
  plt.grid()
  plt.xlabel('$V_{cathode}$ (V)', fontsize='xx-large')
  plt.ylabel('$V_{anode}$ (mV)', fontsize='xx-large')
  plt.title('Figure '+str(k+4) + ' ('+str(name[k])+' nm light, d = 0.3)', fontsize='xx-large')
  plt.annotate('fit V$_s$ st. dev. = %.4f '%perr[-1], (0.52,0.25),
               xycoords='figure fraction',
               fontsize='x-large')
  plt.legend(loc='best',fontsize='large')
  plt.show()
  v_s2.append(params[3])
  print("The stopping voltage is ", params[3].round(4), "+/-", perr[3].round(4), "V")
  print("c ", params[2].round(4), "+/-", perr[2].round(4))
  print("b ", params[1].round(4), "+/-", perr[1].round(4))
  print("a ", params[0].round(4), "+/-", perr[0].round(4))


ii=[1,4,1]
jj=[15,19,18]
name = ['635', '532', '405']
colors = ['r', 'g', 'b']

## "The true retarding potential across the phototube is obtained by adding the
##  magnitudes of the potentials measured at the two BNC output jacks."
x_data_1 = [x_data_r1, x_data_g1, x_data_b1]
x_data_2 = [x_data_r2, x_data_g2, x_data_b2]
y_data_1 = [y_data_r1, y_data_g1, y_data_b1]
y_data_2 = [y_data_r2, y_data_g2, y_data_b2]
v_s1 = []

for k in range(3):
  ## use curve_fit starting with the data at index i and ending at index j
  i=ii[k]
  j=jj[k]

  x1 = x_data_1[k]
  y1 = y_data_1[k]

  params, params_covariance = opt.curve_fit(Eq9_model, x1[i:j], y1[i:j], p0 = [1,1,1,0.4])

  perr = np.sqrt(np.diag(params_covariance))  ## np.diag returns the diagonal elements [0,0] and [1,1]
                                              ## the standard deviation is the square root of the variance

  ## plot the data with the fitted function 
  plt.scatter(x1, y1, label='data points', color=colors[k])  ## plot the data
  ## plot the fit line, also starting at index i and ending at index j
  plt.plot(x1[i:j], Eq9_model(x1[i:j], a=params[0], b=params[1], c=params[2], vs=params[3]), 'black', label='Fit')
  plt.plot(np.full(shape=len(x1), fill_value=params[-1].round(4)), y1, color='gray', label='$V_s$ = %.4f'%params[-1].round(4))
  plt.grid()
  plt.xlabel('$V_{cathode}$ (V)', fontsize='xx-large')
  plt.ylabel('$V_{anode}$ (mV)', fontsize='xx-large')
  plt.title('Figure '+str(k+7) + ' ('+str(name[k])+' nm light, d = 0)', fontsize='xx-large')
  plt.annotate('fit V$_s$ st. dev. = %.4f '%perr[-1], (0.52,0.25),
               xycoords='figure fraction',
               fontsize='x-large')
  plt.legend(loc='best',fontsize='large')
  plt.show()
  v_s1.append(params[3])
  print("The stopping voltage is ", params[3].round(4), "+/-", perr[3].round(4), "V")
  print("c ", params[2].round(4), "+/-", perr[2].round(4))
  print("b ", params[1].round(4), "+/-", perr[1].round(4))
  print("a ", params[0].round(4), "+/-", perr[0].round(4))


## space to make a linear fit of the stopping voltage as a function of incident light frequency

def Eq10toFit(f, h, phi):
  """
  input : 
    f: frequency of light
    h: planck's constant
    phi: workfunction
  output:
    e*v_s = h*f - e*phi
  """
  return h*f - phi

e = 1.60217663e-19
c=299792458
f = [c/(635e-9), c/(532e-9), c/(405e-9)]
params1, params_covariance1 = opt.curve_fit(Eq10toFit, np.array(f)/e, np.array(v_s1), p0 = [1, 1])
params2, params_covariance2 = opt.curve_fit(Eq10toFit, np.array(f)/e, np.array(v_s2), p0 = [1, 1])

perr1 = np.sqrt(np.diag(params_covariance1))
perr2 = np.sqrt(np.diag(params_covariance2))


print(params1, perr1)
print(params2, perr2)


plt.title('Figure 10', fontsize='xx-large')
plt.scatter(f, np.array(v_s1), c='orange', label='data (full I)')
plt.scatter(f, np.array(v_s2), color=(0.5,0.2,0.2), label='data (filtered)')
plt.plot(f, Eq10toFit(np.array(f)/e, params1[0], params1[1]), c='orange', label='fit (full I)       $h$ = %.4f e-34'%(float(params1[0])*10**34))
plt.plot(f, Eq10toFit(np.array(f)/e, params2[0], params2[1]), color=(0.5,0.2,0.2), label='fit (filtered)   $h$ = %.4f e-34'%(float(params2[0])*10**34))
plt.grid()
plt.xlabel('frequency of light (Hz)', fontsize='xx-large')
plt.ylabel('$V_s$ (V)', fontsize='xx-large')
plt.legend()
plt.show()



