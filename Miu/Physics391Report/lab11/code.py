import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import scipy.optimize as opt


#  Output data here
ybco_df = pd.read_csv('YBCO.csv')
bscco_df = pd.read_csv('BSCCO.csv')


def temp(vol, a=525.4425359, b=55647.54346, c=22668.77448):
  return a-np.sqrt(b+c*vol)

ybco_df['resistance'] = ybco_df['voltage']/1000/0.45
bscco_df['resistance'] = bscco_df['voltage']/1000/0.45

ybco_df['thermocouple_vol'] = ybco_df['temperature']
bscco_df['thermocouple_vol'] = bscco_df['temperature']
ybco_df['temperature'] = temp(ybco_df['thermocouple_vol'])
bscco_df['temperature'] = temp(bscco_df['thermocouple_vol'])

plt.figure(figsize=(5,3))
plt.scatter(ybco_df["temperature"], ybco_df['resistance'], s=55, label="YBCO", marker="+", color="darkorange")
plt.plot(ybco_df["temperature"], np.gradient(ybco_df['resistance'],ybco_df["temperature"])*9, alpha=0.3, label="dR/dT", color="blue")
plt.annotate('max dR/dT at T=%.2f K'%ybco_df["temperature"][np.argmax(np.gradient(ybco_df['resistance'],ybco_df["temperature"]))], (0.39,0.25),
             xycoords='figure fraction', fontsize='x-large')
plt.legend()
plt.grid()
plt.title("Figure 1", fontsize=16)
plt.xlabel("Temperature (K)", fontsize=14)
plt.ylabel("Resistance (Omega)", fontsize=14)
plt.show()

plt.figure(figsize=(5,3))
plt.scatter(bscco_df["temperature"], bscco_df['resistance'], s=55, label="BSCCO", marker="+", color="darkorange")
plt.plot(bscco_df["temperature"], np.gradient(bscco_df['resistance'],bscco_df["temperature"])*0.25, alpha=0.3, label="dR/dT", color="blue")
plt.annotate('max dR/dT at T=%.2f K'%bscco_df["temperature"][np.argmax(np.gradient(bscco_df['resistance'],bscco_df["temperature"]))], (0.39,0.3),
             xycoords='figure fraction', fontsize='x-large')
plt.legend()
plt.grid()
plt.title("Figure 2", fontsize=16)
plt.xlabel("Temperature (K)", fontsize=14)
plt.ylabel("Resistance (Omega)", fontsize=14)
plt.show()
