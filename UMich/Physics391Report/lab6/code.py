import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import scipy.optimize as opt
import os
import sys




## Define relevant experimental constants here
c = 2.9979e8
e = 1.6022e-19
h = 6.62607015e-34
me = 9.1093837e-31
mp = 1.67262192e-27


# Read in data from your data directory here.  Use pd.read_csv.
Hyd_df = pd.read_csv('data/Hydrogen.csv')
Da1_df = pd.read_csv('data/Da_1.csv')
Da2_df = pd.read_csv('data/Da_2.csv')
Ha1_df = pd.read_csv('data/Ha_1.csv')
Ha2_df = pd.read_csv('data/Ha_2.csv')
Hydt_df = pd.read_csv('data/Hydrogen_t.csv')


#  Make plots here
lambs_all = Hyd_df['Run 1: Wavelength (nm)']

name = ['a','b','c','d','e','f']

for i in range(6):
  plt.plot(lambs_all, Hyd_df['Run '+str(i+1)+': Intensity (rel)'], c='red')
  plt.grid()
  plt.title('Figure 1'+name[i], fontsize='xx-large')
  plt.ylabel(r'Intensity', fontsize='xx-large')
  plt.xlabel('Wavelength (nm)', fontsize='xx-large')
  plt.show()


plt.plot(Hydt_df['Run 1: Wavelength (nm)'], Hydt_df['Run 1: Intensity (rel)'], c='red')
plt.plot(Hydt_df['Run 3: Wavelength (nm)'], Hydt_df['Run 3: Intensity (rel)'], c='red')
plt.plot(Hydt_df['Run 4: Wavelength (nm)'], Hydt_df['Run 4: Intensity (rel)'], c='red')
plt.plot(Hydt_df['Run 5: Wavelength (nm)'], Hydt_df['Run 5: Intensity (rel)'], c='red')
plt.plot(Hydt_df['Run 6: Wavelength (nm)'], Hydt_df['Run 6: Intensity (rel)'], c='red')
plt.grid()
plt.title('Figure 1', fontsize='xx-large')
plt.ylabel(r'Intensity', fontsize='xx-large')
plt.xlabel('Wavelength (nm)', fontsize='xx-large')
plt.show()


def calculate_mean_peak(wavelength, intensity) :
  '''Calculates the mean peak intensity via Eqn 7
  '''
  return np.sum([x*y for (x,y) in zip(wavelength, intensity)])/np.sum(intensity)
  #### FILL THIS OUT TO RETURN THE MEAN WAVELENGTH ####

alpha = calculate_mean_peak(Hydt_df['Run 1: Wavelength (nm)'], Hydt_df['Run 1: Intensity (rel)'])
beta = calculate_mean_peak(Hydt_df['Run 3: Wavelength (nm)'], Hydt_df['Run 3: Intensity (rel)'])
delta = calculate_mean_peak(Hydt_df['Run 4: Wavelength (nm)'], Hydt_df['Run 4: Intensity (rel)'])
gamma =  calculate_mean_peak(Hydt_df['Run 5: Wavelength (nm)'], Hydt_df['Run 5: Intensity (rel)'])
epsilon = calculate_mean_peak(Hydt_df['Run 6: Wavelength (nm)'][:11], Hydt_df['Run 6: Intensity (rel)'][:11])
zeta = calculate_mean_peak(Hydt_df['Run 7: Wavelength (nm)'][:11], Hydt_df['Run 7: Intensity (rel)'][:11])


def gaussian_model(x, a, b, xp, sigma) :
  '''Gaussian model
  '''
  return a + b * np.exp(-(x-xp)**2/(2*sigma**2))



### OPTIONAL SPACE TO FIT EACH CURVE HERE AND IN CELLS BELOW TO EXTRACT PEAK LOCATIONS ###
params, params_covariance = opt.curve_fit(gaussian_model, Hydt_df['Run 1: Wavelength (nm)'], Hydt_df['Run 1: Intensity (rel)'], p0 = [0.01,0.01,650,1])
perr = np.sqrt(np.diag(params_covariance))

# Calculate each peak here, and comparison to known values
err_alpha = np.abs(alpha-656.27)/656.27
err_beta = np.abs(beta-486.13)/486.13
err_delta = np.abs(delta-434.05)/434.05
err_gamma = np.abs(gamma-410.17)/410.17
err_epsilon = np.abs(epsilon-397.01)/397.01
err_zeta = np.abs(zeta-388.91)/388.91


# Make plot here and/or in a new set of cells below.
n=np.array([3,4,5,6,7,8])
n_x = np.array([(1/4)-(1/(i*i)) for i in n])
lambs = np.array([alpha, beta, delta, gamma, epsilon, zeta])/1000
lambs_y = 1/lambs

plt.scatter(n_x, 1/lambs, c='red')
plt.grid()
plt.title('Figure 2', fontsize='xx-large')
plt.ylabel(r'1/$\lambda$ [microns$^{-1}$]', fontsize='xx-large')
plt.xlabel(r'$\frac{1}{4} - \frac{1}{n}$', fontsize='xx-large')
plt.show()


def linear_model(x, m, b) :
  '''
  x - list, x-dataset
  m - float, slope
  b - float, y-intercept
  '''
  return m*x+b


# Perform your fit
popt, pcov = opt.curve_fit(linear_model, n_x, lambs_y, np.array([1,1]))
sig_m = np.sqrt(pcov[0])



# Overplot data and best fit model using plt.plot and plt.scatter,
plt.scatter(n_x, 1/lambs, c='red', label = 'data')
plt.plot(n_x, linear_model(n_x, popt[0], popt[1]), label = 'linear fit')
plt.grid()
plt.title('Figure 3', fontsize='xx-large')
plt.legend()
plt.annotate('m = %.4f '%popt[0], (0.63,0.35),
               xycoords='figure fraction',
               fontsize='x-large')
plt.annotate('b = %.4f'%popt[1], (0.63,0.28),
               xycoords='figure fraction',
               fontsize='x-large')
plt.annotate(r'$\sigma_b$ = %.4f'%sig_m[0], (0.63,0.21),
               xycoords='figure fraction',
               fontsize='x-large')
plt.ylabel(r'1/$\lambda$ [microns$^{-1}$]', fontsize='xx-large')
plt.xlabel(r'$\frac{1}{4} - \frac{1}{n}$', fontsize='xx-large')
plt.show()


### READ IN CSV FILES HERE AS DATAFRAMES ###
da = pd.read_csv('data/Da_1.csv')
da = pd.concat([da, pd.read_csv('data/Da_2.csv')], axis=1)
da.columns = ['L1', 'I1',
              'L2', 'I2',
              'L3', 'I3',
              'L4', 'I4',
              'L5', 'I5',
              'L6', 'I6',
              'L7', 'I7',
              'L8', 'I8',
              'L9', 'I9',
              'L10', 'I10',]

ha = pd.read_csv('data/Ha_1.csv')
ha = pd.concat([ha, pd.read_csv('data/Ha_2.csv')], axis=1)
ha.columns = ['L1', 'I1',
              'L2', 'I2',
              'L3', 'I3',
              'L4', 'I4',
              'L5', 'I5',
              'L6', 'I6',
              'L7', 'I7',
              'L8', 'I8',
              'L9', 'I9',
              'L10', 'I10',]

da1 = da[['L1', 'I1', 'L2', 'I2', 'L3', 'I3', 'L4', 'I4', 'L5', 'I5']]
da2 = da[['L6', 'I6', 'L7', 'I7', 'L8', 'I8', 'L9', 'I9', 'L10', 'I10']]
ha1 = ha[['L1', 'I1', 'L2', 'I2', 'L3', 'I3', 'L4', 'I4', 'L5', 'I5']]
ha2 = ha[['L6', 'I6', 'L7', 'I7', 'L8', 'I8', 'L9', 'I9', 'L10', 'I10']]


# Create your summed/averaged spectra here
da1_avgI = da1['I1'].values
ha1_avgI = ha1['I1'].values
for i in range(2,6):
  da1_avgI = da1_avgI + da1['I'+str(i)].values
  ha1_avgI = ha1_avgI + ha1['I'+str(i)].values
da1_avgI = da1_avgI/5
ha1_avgI = ha1_avgI/5

da2_avgI = da2['I6'].values
ha2_avgI = ha2['I6'].values
for i in range(7,11):
  da2_avgI = da2_avgI + da2['I'+str(i)].values
  ha2_avgI = ha2_avgI + ha2['I'+str(i)].values
da2_avgI = da2_avgI/5
ha2_avgI = ha2_avgI/5


# Plots here
plt.plot(da1['L1'], da1_avgI, c='red', label = 'Deuterium data')
plt.plot(da1['L1'], ha1_avgI, c='blue', label = 'Helium data')
plt.grid()
plt.title('Figure 4a', fontsize='xx-large')
plt.legend()
plt.ylabel(r'Intensity (scaled to 1)', fontsize='xx-large')
plt.xlabel(r'wavelength [nm]', fontsize='xx-large')
plt.show()


plt.plot(da1['L1'], da2_avgI, c='red', label = 'Deuterium data')
plt.plot(da1['L1'], ha2_avgI, c='blue', label = 'Helium data')
plt.grid()
plt.title('Figure 4b', fontsize='xx-large')
plt.legend()
plt.ylabel(r'Intensity (scaled to 1)', fontsize='xx-large')
plt.xlabel(r'wavelength [nm]', fontsize='xx-large')
plt.show()


#  Fit for delta_h from Experiments hydrogen and deuterium data
def gaussian_model(x, a, b, xp, sigma) :
  '''Gaussian model
  '''
  return a + b * np.exp(-(x-xp)**2/(2*sigma**2))


lambs = da1['L1']
parD1, corD1 = opt.curve_fit(gaussian_model, lambs, da1_avgI, p0 = [0.1,0.1,660,10])
sigD1 = np.sqrt(np.diag(corD1))
del_D1 = parD1[2]-656

parD2, corD2 = opt.curve_fit(gaussian_model, lambs, da2_avgI, p0 = [0.1,0.1,660,10])
sigD2 = np.sqrt(np.diag(corD2))
del_D2 = parD2[2]-656

parH1, corH1 = opt.curve_fit(gaussian_model, lambs, ha1_avgI, p0 = [0.1,0.1,656,10])
sigH1 = np.sqrt(np.diag(corH1))
del_H1 = parH1[2]-656

parH2, corH2 = opt.curve_fit(gaussian_model, lambs, ha2_avgI, p0 = [0.1,0.1,656,10])
sigH2 = np.sqrt(np.diag(corH2))
del_H2 = parH2[2]-656

expt1_sig = np.sqrt(sigD1[2]**2 + sigH1[2]**2)
expt2_sig = np.sqrt(sigD2[2]**2 + sigH2[2]**2)

print(del_D1,del_D2)
print(del_H1,del_H2)

print(parH1)

#  Overplot best-fit model and data for experiments
plt.scatter(da1['L1'], da1_avgI, c='red', label = 'Deuterium data')
plt.scatter(da1['L1'], ha1_avgI, c='blue', label = 'Helium data')
plt.plot(np.linspace(da1['L1'][0], da1['L1'].values.tolist()[-1], 100),
         gaussian_model(np.linspace(da1['L1'][0], da1['L1'].values.tolist()[-1], 100),
                        parD1[0], parD1[1], parD1[2], parD1[3]),
         c='red', label = 'Deuterium fit')
plt.plot(np.linspace(da1['L1'][0], da1['L1'].values.tolist()[-1], 100),
         gaussian_model(np.linspace(da1['L1'][0], da1['L1'].values.tolist()[-1], 100),
                        parH1[0], parH1[1], parH1[2], parH1[3]),
         c='blue', label = 'Deuterium fit')
plt.annotate(r'$\delta_H$ = %.4f'%del_H1, (0.68,0.28),
              xycoords='figure fraction',
              fontsize='x-large')
plt.annotate(r'$\delta_D$ = %.4f'%del_D1, (0.68,0.33),
              xycoords='figure fraction',
              fontsize='x-large')
plt.annotate(r'$\sigma$ = %.4f'%expt1_sig, (0.68,0.38),
              xycoords='figure fraction',
              fontsize='x-large')
plt.grid()
plt.title('Figure 5a', fontsize='xx-large')
plt.legend()
plt.ylabel(r'Intensity (scaled to 1)', fontsize='xx-large')
plt.xlabel(r'wavelength [nm]', fontsize='xx-large')
plt.show()


plt.scatter(da1['L1'], da2_avgI, c='red', label = 'Deuterium data')
plt.scatter(da1['L1'], ha2_avgI, c='blue', label = 'Helium data')
plt.plot(np.linspace(da1['L1'][0], da1['L1'].values.tolist()[-1], 100),
         gaussian_model(np.linspace(da1['L1'][0], da1['L1'].values.tolist()[-1], 100),
                        parD2[0], parD2[1], parD2[2], parD2[3]),
         c='red', label = 'Deuterium fit')
plt.plot(np.linspace(da1['L1'][0], da1['L1'].values.tolist()[-1], 100),
         gaussian_model(np.linspace(da1['L1'][0], da1['L1'].values.tolist()[-1], 100),
                        parH2[0], parH2[1], parH2[2], parH2[3]),
         c='blue', label = 'Deuterium fit')
plt.annotate(r'$\delta_H$ = %.4f'%del_H2, (0.68,0.28),
              xycoords='figure fraction',
              fontsize='x-large')
plt.annotate(r'$\delta_D$ = %.4f'%del_D2, (0.68,0.33),
              xycoords='figure fraction',
              fontsize='x-large')
plt.annotate(r'$\sigma$ = %.4f'%expt2_sig, (0.68,0.38),
              xycoords='figure fraction',
              fontsize='x-large')
plt.grid()
plt.title('Figure 5b', fontsize='xx-large')
plt.legend()
plt.ylabel(r'Intensity (scaled to 1)', fontsize='xx-large')
plt.xlabel(r'wavelength [nm]', fontsize='xx-large')
plt.show()

## NOTE: You may need to change the name of the dataframe if this is not the variable name you used
##   for df_hydrogen_expt1.
h1d1_avgI = np.concatenate((ha1_avgI, da1_avgI))
h2d2_avgI = np.concatenate((ha2_avgI, da2_avgI))

Len_ha = ha.shape[0]


def two_gaussian_model(wavelengths, aH, bH, lam_H, aD, bD, lam_D, sigma,
                       length_hydrogen=Len_ha) :
  ''' Model to fit both
  '''

  hyd_gaussian = gaussian_model(wavelengths[:length_hydrogen], aH, bH, lam_H, sigma)
  deut_gaussian = gaussian_model(wavelengths[length_hydrogen:], aD, bD, lam_D, sigma)

  return np.append(hyd_gaussian, deut_gaussian)





#  Perform your fit here
parj1, corj1 = opt.curve_fit(two_gaussian_model, lambs.append(lambs).values, h1d1_avgI, p0 = [0.01,0.01, 655,0.01, 0.01, 655,20])
sigj1 = np.sqrt(np.diag(corj1))



parj2, corj2 = opt.curve_fit(two_gaussian_model, lambs.append(lambs).values, h2d2_avgI, p0 = [0.01,0.01,655,0.01,0.01, 655, 20])
sigj2 = np.sqrt(np.diag(corj2))

lam_space = np.linspace(da1['L1'][0], da1['L1'].values.tolist()[-1], 100)
lam_spaces = np.concatenate((lam_space, lam_space))
y_1 = two_gaussian_model(lam_spaces, parj1[0], parj1[1], parj1[2], parj1[3], parj1[4], parj1[5], parj1[6], 100)
y_2 = two_gaussian_model(lam_spaces, parj2[0], parj2[1], parj2[2], parj2[3], parj2[4], parj2[5], parj2[6], 100)

del_H1_j = parj1[2]-656
del_D1_j = parj1[5]-656
del_H2_j = parj2[2]-656
del_D2_j = parj2[5]-656


expt1_sig_j = np.sqrt(sigj1[2]**2 + sigj1[5]**2)
expt2_sig_j = np.sqrt(sigj2[2]**2 + sigj2[5]**2)

plt.scatter(da1['L1'], da1_avgI, c='red', label = 'Deuterium data')
plt.scatter(da1['L1'], ha1_avgI, c='blue', label = 'Helium data')
plt.plot(lam_spaces[100:], y_1[100:], c='red', label = 'Deuterium fit')
plt.plot(lam_spaces[:100], y_1[:100], c='blue', label = 'Helium fit')
plt.annotate(r'$\delta_H$ = %.4f'%del_H1_j, (0.68,0.28),
              xycoords='figure fraction',
              fontsize='x-large')
plt.annotate(r'$\delta_D$ = %.4f'%del_D1_j, (0.68,0.33),
              xycoords='figure fraction',
              fontsize='x-large')
plt.annotate(r'$\sigma$ = %.4f'%expt1_sig_j, (0.68,0.38),
              xycoords='figure fraction',
              fontsize='x-large')
plt.grid()
plt.title('Figure 5a', fontsize='xx-large')
plt.legend()
plt.ylabel(r'Intensity (scaled to 1)', fontsize='xx-large')
plt.xlabel(r'wavelength [nm]', fontsize='xx-large')
plt.show()


plt.scatter(da1['L1'], da2_avgI, c='red', label = 'Deuterium data')
plt.scatter(da1['L1'], ha2_avgI, c='blue', label = 'Helium data')
plt.plot(lam_spaces[:100], y_2[:100], c='blue', label = 'Helium fit')
plt.plot(lam_spaces[100:], y_2[100:], c='red', label = 'Deuterium fit')
plt.annotate(r'$\delta_H$ = %.4f'%del_H2_j, (0.68,0.28),
              xycoords='figure fraction',
              fontsize='x-large')
plt.annotate(r'$\delta_D$ = %.4f'%del_D2_j, (0.68,0.33),
              xycoords='figure fraction',
              fontsize='x-large')
plt.annotate(r'$\sigma$ = %.4f'%expt2_sig_j, (0.68,0.38),
              xycoords='figure fraction',
              fontsize='x-large')
plt.grid()
plt.title('Figure 5b', fontsize='xx-large')
plt.legend()
plt.ylabel(r'Intensity (scaled to 1)', fontsize='xx-large')
plt.xlabel(r'wavelength [nm]', fontsize='xx-large')
plt.show()


# Calculate here
c = 299792458


f_H1_j = c/parj1[2]
f_D1_j = c/parj1[5]
del_f_1j = f_H1_j - f_D1_j
ratio_1j = (f_H1_j - f_D1_j)/f_H1_j*2
sigfD1j = -c*sigj1[2]/(parj1[2]**2)
sigfH1j = -c*sigj1[5]/(parj1[5]**2)
sig_rj1 = np.sqrt((np.sqrt(sigfD1j**2 + sigfD1j**2)/del_f_1j)**2 + (sigfH1j/f_H1_j)**2)



f_H2_j = c/parj2[2]
f_D2_j = c/parj2[5]
del_f_2j = f_H2_j - f_D2_j
ratio_2j = (f_H2_j - f_D2_j)/f_H2_j*2
sigfD2j = -c*sigj2[2]/(parj2[2]**2)
sigfH2j = -c*sigj2[5]/(parj2[5]**2)
sig_rj2 = np.sqrt((np.sqrt(sigfD2j**2 + sigfD2j**2)/del_f_2j)**2 + (sigfH2j/f_H2_j)**2)


f_D1 = c/parD1[2]
f_H1 = c/parH1[2]
del_f_1 = f_H1 - f_D1
ratio_1 = (f_H1 - f_D1)/f_H1*2
sigfD1 = -c*sigD1[2]/(parD1[2]**2)
sigfH1 = -c*sigH1[2]/(parH1[2]**2)
sig_r1 = np.sqrt((np.sqrt(sigfD1**2 + sigfD1**2)/del_f_1)**2 + (sigfH1/f_H1)**2)


f_D2 = c/parD2[2]
f_H2 = c/parH2[2]
del_f_2 = f_H2 - f_D2
ratio_2 = (f_H2 - f_D2)/f_H2*2
sigfD2 = -c*sigD2[2]/(parD2[2]**2)
sigfH2 = -c*sigH2[2]/(parH2[2]**2)
sig_r2 = np.sqrt((np.sqrt(sigfD2**2 + sigfD2**2)/del_f_2)**2 + (sigfH2/f_H2)**2)


print(parj1[2]-parj1[5], parj2[2]-parj2[5], parH1[2]-parD1[2], parH2[2]-parD2[2])
print(ratio_1j, ratio_2j, ratio_1, ratio_2)
print(sig_rj1, sig_rj2, sig_r1, sig_r2)
