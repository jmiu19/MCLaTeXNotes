import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import scipy.optimize as opt
import os

# Read in the three spectral intensity datasets from your data directory here.  Use pd.read_csv
speIntB1_df = pd.read_csv('data/Spectral_Intensity_Data_black1.txt',
                          skiprows=0, delimiter='\t').drop(17)
speIntB2_df = pd.read_csv('data/Spectral_Intensity_Data_black2.txt',
                          skiprows=0, delimiter='\t').drop(17)
speIntR1_df = pd.read_csv('data/Spectral_Intensity_Data_red1.txt',
                          skiprows=0, delimiter='\t').drop(17)
speIntG1_df = pd.read_csv('data/Spectral_Intensity_Data_green1.txt',
                          skiprows=0, delimiter='\t').drop(17)
radPower_df = pd.read_csv('data/Radiated_Power_Data.txt', skiprows=0,
                          delimiter='\t').drop(43)

# Define and calculate resistance constants, filter resistances, and resitance ratios

# resistance values here
r_cords = 0.028
r_filament_and_cords = 5.129
r_roomT = r_filament_and_cords - r_cords
r_filament_and_cords_rP = 5.135
r_roomTr = r_filament_and_cords_rP - r_cords

# Filter resistance and ratios
green_resistance = [V/I for V,I in zip(speIntG1_df['Voltage'], speIntG1_df['Current'])]
red_resistance = [V/I for V,I in zip(speIntR1_df['Voltage'], speIntR1_df['Current'])]
dark_resistance = [V/I for V,I in zip(speIntB2_df['Voltage'], speIntB2_df['Current'])]
radP_resistance = [V/I for V,I in zip(radPower_df['Voltage'], radPower_df['Current'])]


green_resistance_ratio = [r/r_roomT for r in green_resistance]
red_resistance_ratio = [r/r_roomT for r in red_resistance]
dark_resistance_ratio = [r/r_roomT for r in dark_resistance]
radP_resistance_ratio = [r/r_roomTr for r in radP_resistance]

def calculate_temperature(r, a=-4.129538e-1, b=4.360552e-3,
                            c=7.399998e-7, d=6.195380e-5) :
  '''Using fit parameters for the resistivity ratio of tungsten as a function of temperature,
  we calculate the temperature as a function of the resistance ratio.
  Input:
  r (float): resistance ratio with unity at 300K

  Output:
  temperature (float) : Temperature to convert values for lamp voltage and current to resistance
  '''
  return (d*r-b+np.sqrt((d*r-b)**2 + 4*(r-a)*c))/(2*c)


# Calculate temperatures for each filter

green_temperature = np.array([calculate_temperature(r) for r in green_resistance_ratio])
red_temperature = np.array([calculate_temperature(r) for r in red_resistance_ratio])
dark_temperature = np.array([calculate_temperature(r) for r in dark_resistance_ratio])
radP_temperature = np.array([calculate_temperature(r) for r in radP_resistance_ratio])

# Plot log10I_photodiode vs. 1/T for all three filters
bT = [1000/t for t in dark_temperature]
bI = [np.log10(I/np.max(speIntB2_df['PhotoCurrent'].values))
      for I in speIntB2_df['PhotoCurrent']]

rT = [1000/t for t in red_temperature]
rI = [np.log10(I/np.max(speIntR1_df['PhotoCurrent'].values))
      for I in speIntR1_df['PhotoCurrent']]

gT = [1000/t for t in green_temperature]
gI = [np.log10(I/np.max(speIntG1_df['PhotoCurrent'].values))
      for I in speIntG1_df['PhotoCurrent']]

plt.scatter(bT,bI, c='black', label="black filter",alpha=0.7)
plt.scatter(rT,rI, c='red', label="red filter",alpha=0.7)
plt.scatter(gT,gI, c='green', label="green filter",alpha=0.7)
plt.ylabel("log$_{10}$(i/i$_{max}$)",fontsize=16)
plt.xlabel("1000/T",fontsize=16)
plt.title('Figure 2',fontsize=16)
plt.legend()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

## comparing two datasets
plt.scatter(speIntB1_df['Time'], speIntB1_df['Current'], label='1st dataset')
plt.scatter(speIntB2_df['Time'], speIntB2_df['Current'], label='2nd dataset')
plt.ylabel("Current [A]",fontsize=16)
plt.xlabel("Time [s]",fontsize=16)
plt.title('Figure 7',fontsize=16)
plt.legend()
plt.show()

plt.scatter(speIntB1_df['Time'], speIntB1_df['Voltage'], label='1st dataset')
plt.scatter(speIntB2_df['Time'], speIntB2_df['Voltage'], label='2nd dataset')
plt.ylabel("Voltage [V]",fontsize=16)
plt.xlabel("Time [s]",fontsize=16)
plt.title('Figure 8',fontsize=16)
plt.legend()
plt.show()

plt.scatter(speIntB1_df['Time'], speIntB1_df['PhotoCurrent'], label='1st dataset')
plt.scatter(speIntB2_df['Time'], speIntB2_df['PhotoCurrent'], label='2nd dataset')
plt.ylabel("Photodiode Current [A]",fontsize=16)
plt.xlabel("Time [s]",fontsize=16)
plt.title('Figure 9',fontsize=16)
plt.legend()
plt.show()
plt.show()


# Plot I_photodiode vs. T for all three filters
plt.scatter(dark_temperature, speIntB2_df['PhotoCurrent'].values,
            c='black', label="dark filter",alpha=0.7)
plt.scatter(red_temperature, speIntR1_df['PhotoCurrent'].values,
            c='red', label="red filter",alpha=0.7)
plt.scatter(green_temperature, speIntG1_df['PhotoCurrent'].values,
            c='green', label="green filter",alpha=0.7)
plt.ylabel("Photodiode Current (A)",fontsize=16)
plt.xlabel("Temperature (K)",fontsize=16)
plt.title('Figure 1',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend()
plt.show()

def planck_model_to_fit(temperature, wavelength_param) :
  '''Model to fit the intensity vs. temperature model to
  Inputs:
  temperature (array-like): temperature measured, calculated from resistivity ratio data
  wavelength_param (float): wavelength of filter, can be fit

  Outputs:
  log_norm_current (array-like): log of the current normalized by the max
  '''

  h = 6.626070040e-34
  c = 2.99792458e8
  k = 1.38064852e-23

  term1 = np.log10(np.exp(h*c/wavelength_param/k/np.max(temperature)) - 1)
  term2 = np.log10(np.exp(h*c/wavelength_param/k/temperature) - 1)

  log_norm_current = term1 - term2

  return log_norm_current

def plot_iteration_of_least_squares(temperature, log_current_norm_data,
                                    log_current_norm_model) :
  ''''''

  plt.plot(1000/temperature,log_current_norm_data)
  plt.plot(1000/temperature,log_current_norm_model, ls=':')
  plt.xlabel("1000/T",fontsize=16)
  plt.ylabel("log$_{10}$(i/i$_{max}$)",fontsize=16)
  plt.show()

def planck_chisq_to_minimize(wavelength_param, temperature, current, plot=True) :
  '''Chi squared to minimize when fitting the current vs.
     temperature data to the Planck spectrum
  Inputs:
  wavelength_param (float): wavelength of filter
                            divided by the speed of light, can be fit
  temperature (array-like): temperature measured,
                            calculated from resistivity ratio data
  current (array-like): current measured, calculated from resistivity ratio data
  plot (boolean, optional): option to plot the model and data with each iteration

  Outputs:
  least_sq (float): value of the least square value
  '''

  log_current_norm_data = np.log10(current/np.max(current))
  log_current_norm_model = planck_model_to_fit(temperature, wavelength_param)

  chi_squared_of_model = np.sum((log_current_norm_data-log_current_norm_model)**2)

  least_sq=chi_squared_of_model

  if plot :
    plot_iteration_of_least_squares(temperature, log_current_norm_data,
                                    log_current_norm_model)
    print("Least squares this iteration:%f.2 nm"%least_sq)

  return least_sq

# Fit points to Eqn 10
wavelengths = [500e-9 +i*10e-11 for i in range(0,10000)]
photo_Is = [speIntG1_df['PhotoCurrent'].values,
            speIntR1_df['PhotoCurrent'].values,
            speIntB1_df['PhotoCurrent'].values]
temps = [green_temperature,
         red_temperature,
         dark_temperature]
fit_lams = []
opt_lams = []
covars = []
chis = []

# iterate all three colors to find best fit or optimized lambda
for i in [0,1,2]:
  # extract parameters for this color
  chi = []
  temp, photo_I = temps[i], photo_Is[i]
  # find best fit lambda
  for lam in wavelengths:
    chi.append(planck_chisq_to_minimize(lam, temp, photo_I, False))
  min_chi_index = chi.index(np.min(chi))
  chis.append(np.min(chi))
  fit_lams.append(wavelengths[min_chi_index])
  # find optimized lambda
  log_current_norm_data = np.log10(photo_I/np.max(photo_I))
  opt_lam, covariance = opt.curve_fit(planck_model_to_fit,
                                      temp,
                                      log_current_norm_data,
                                      p0 = [550e-9])
  opt_lams.append(opt_lam)
  covars.append(covariance)

# Overplotted best-fit model with data points here
colors = [['green', (0.3,1,0.4), (0.2,0.6,0.2)],
          ['red', (1,0.3,0.4), (0.6,0.2,0.2)],
          ['black', (0,0,0), (0.3,0.3,0.3)]]
print_cor = ['G', 'R', 'D']
for i in [0,1,2]:
  temp, photo_I = temps[i], photo_Is[i]
  log_I_norm_data = np.log10(photo_I/np.max(photo_I))+i*0.5
  fit_lam, opt_lam, covar = fit_lams[i], opt_lams[i], covars[i]
  log_current_norm_fit = planck_model_to_fit(temp, fit_lam)+i*0.5
  log_current_norm_opt = planck_model_to_fit(temp, opt_lam)+i*0.5
  plt.plot(1000/temp,log_I_norm_data, ls='none', marker='o', c=colors[i][0])
  plt.plot(1000/temp,log_current_norm_fit, ls='-.', c=colors[i][1])
  plt.plot(1000/temp,log_current_norm_opt, ls=':', c=colors[i][2])
  lam_print = fit_lam*10**9
  plt.annotate(print_cor[i]+' fit $\lambda$=%.2f nm'%lam_print, (0.15,0.2+i*0.06),
               xycoords='figure fraction',
               fontsize='x-large')

plt.xlabel("1000/T",fontsize=16)
plt.ylabel("$0.5*i + $log$_{10}$(i/i$_{max}$)",fontsize=16)
plt.title('Figure 4', fontsize=16)
plt.show()

print(covars)
print(opt_lams)
print(fit_lams)


# Read in the radiated power data from your data directory here.  Use pd.read_csv
radPower_df['Temp'] = radP_temperature
radPower_df['R/R0'] = radP_resistance_ratio
radPower_df['R'] = radP_resistance
radPower_df['Power'] = [I*V for I,V in zip(radPower_df['Current'],
                                           radPower_df['Voltage'])]


# Plot log P vs. log T
plt.plot(np.log10(radPower_df['Temp'].values) ,
         np.log10(radPower_df['Power'].values),
         ls='none', marker='o', c=colors[i][0])
plt.xlabel("log(Temp)",fontsize=16)
plt.ylabel("log(Power)",fontsize=16)
plt.title('Figure 3',fontsize=16)
plt.show()


def linearFit(x, m, b):
  """
  input:
    x: x-data to be fitted
    m: slope of the linear fit
    b: y-interception of the linear fit
  return:
    y: y=mx+b
  """
  return m*x+b


## linear fit the power law
opt_mb, covariance = opt.curve_fit(linearFit,
                                   np.log10(radPower_df['Temp'].values),
                                   np.log10(radPower_df['Power'].values),
                                   p0 = [4,0])
## linear fit but taking out the first few outliers
opt_mb_o, covariance_o = opt.curve_fit(linearFit,
                                       np.log10(radPower_df['Temp'].values[4:]),
                                       np.log10(radPower_df['Power'].values[4:]),
                                       p0 = [4,0])

opt_m, opt_b = opt_mb[0], opt_mb[1]
opt_m_o, opt_b_o = opt_mb_o[0], opt_mb_o[1]

print(opt_m, opt_b)
print(opt_m_o, opt_b_o)

## plot the linear fit of the power law
plt.plot(np.log10(radPower_df['Temp'].values),
         np.log10(radPower_df['Power'].values),
         ls='none', marker='o', c='black', alpha=0.8)
plt.plot(np.log10(radPower_df['Temp'].values) ,
         linearFit(np.log10(radPower_df['Temp'].values),
                   opt_m, opt_b),
         ls='-', c='r')
plt.plot(np.log10(radPower_df['Temp'].values) ,
         linearFit(np.log10(radPower_df['Temp'].values),
                   opt_m_o, opt_b_o),
         ls='-', c='b')
plt.xlabel("log(Temp)",fontsize=16)
plt.ylabel("log(Power)",fontsize=16)
plt.annotate('red fit m = %.2f '%opt_m, (0.65,0.2),
             xycoords='figure fraction',
             fontsize='x-large')
plt.annotate('blu fit m = %.2f '%opt_m_o, (0.65,0.25),
             xycoords='figure fraction',
             fontsize='x-large')
plt.annotate('blu m var = %.3f '%covariance_o[0][0], (0.15,0.81),
             xycoords='figure fraction',
             fontsize='x-large')
plt.annotate('blu b var = %.3f '%covariance_o[1][1], (0.15,0.71),
             xycoords='figure fraction',
             fontsize='x-large')
plt.annotate('red m var = %.3f '%covariance[0][0], (0.15,0.86),
             xycoords='figure fraction',
             fontsize='x-large')
plt.annotate('red b var = %.3f '%covariance[1][1], (0.15,0.76),
             xycoords='figure fraction',
             fontsize='x-large')
plt.title('Figure 5', fontsize=16)
plt.show()





def stefBoltzFit(x, sig, alp):
  """
  input:
    x: x-data to be fitted
    sig: stefan-boltzmann constant
    alp: power of the temperature
  return:
    y: sigma*x**(alpha)
  """
  return sig*x**(alp)

# power fit the power law
opt_sa, covariance_sa = opt.curve_fit(stefBoltzFit,
                                      radPower_df['Temp'].values,
                                      radPower_df['Power'].values,
                                      p0 = [0,4])
# power fit but taking out the first few outliers
opt_sa_o, covariance_sa_o = opt.curve_fit(stefBoltzFit,
                                          radPower_df['Temp'].values[4:],
                                          radPower_df['Power'].values[4:],
                                          p0 = [0,4])



opt_s, opt_a = opt_sa[0], opt_sa[1]
opt_s_o, opt_a_o = opt_sa_o[0], opt_sa_o[1]

print(opt_s, opt_a)
print(opt_s_o, opt_a_o)

## plot the power fit
plt.plot(radPower_df['Temp'].values,
         radPower_df['Power'].values,
         ls='none', marker='o', c='black', alpha=0.8)
plt.plot(radPower_df['Temp'].values,
         stefBoltzFit(radPower_df['Temp'].values,
                      opt_s, opt_a),
         ls='-', c='r')
plt.plot(radPower_df['Temp'].values,
         stefBoltzFit(radPower_df['Temp'].values,
                      opt_s_o, opt_a_o),
         ls='-', c='b')
plt.xlabel("Temp [K]",fontsize=16)
plt.ylabel("Power [W]",fontsize=16)
plt.annotate('red fit alpha = %.2f '%opt_a, (0.15,0.5),
             xycoords='figure fraction',
             fontsize='x-large')
plt.annotate('blu fit alpha = %.2f '%opt_a_o, (0.15,0.55),
             xycoords='figure fraction',
             fontsize='x-large')
plt.annotate('blu alpha var = %.3f '%covariance_sa_o[0][0], (0.15,0.80),
             xycoords='figure fraction',
             fontsize='x-large')
plt.annotate('blu $\sigma_s$ var = %.3f '%covariance_sa_o[1][1], (0.15,0.70),
             xycoords='figure fraction',
             fontsize='x-large')
plt.annotate('red alpha var = %.3f '%covariance_sa[0][0], (0.15,0.85),
             xycoords='figure fraction',
             fontsize='x-large')
plt.annotate('red $\sigma_s$ var = %.3f '%covariance_sa[1][1], (0.15,0.75),
             xycoords='figure fraction',
             fontsize='x-large')
plt.title('Figure 6', fontsize=16)
plt.legend()
plt.show()
