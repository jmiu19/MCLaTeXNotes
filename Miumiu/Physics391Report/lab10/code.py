import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

import os            ##  This module is for "operating system" interfaces
import sys           ##  This module is for functionality relevant to the python run time

GOOGLE_PATH_AFTER_MYDRIVE = 'Physics391/Lab10HallEffectSemiconductor'
GOOGLE_DRIVE_PATH = os.path.join('drive','My Drive', GOOGLE_PATH_AFTER_MYDRIVE)
print(os.listdir(GOOGLE_DRIVE_PATH))

# Append the directory path of this notebook to what python easily "sees"
sys.path.append(GOOGLE_DRIVE_PATH)

# Make your current working directory the directory path of this notebookand data
os.chdir(GOOGLE_DRIVE_PATH)

hall_head = pd.read_csv("hallEffect_head.csv")
hall_tail = pd.read_csv("hallEffect_tail.csv")
temperature = pd.read_csv("temperature.csv")

plt.figure(figsize=(5,3))
plt.scatter(hall_head["Current (mA)"], hall_head["Voltage (V)"], s=35, label="Voltage (V)", marker="3", color="orange")
plt.scatter(hall_head["Current (mA)"], hall_head["Hall Voltage (V)"], s=10, label="Hall Voltage (V)", color="black")
plt.legend()
plt.grid()
plt.title("Outward B field", fontsize=16)
plt.xlabel("Current (mA)", fontsize=14)
plt.ylabel("Voltage (V)", fontsize=14)

plt.figure(figsize=(5,3))
plt.scatter(hall_tail["Current (mA)"], hall_tail["Voltage (V)"], s=35, label="Voltage (V)", marker="3", color="orange")
plt.scatter(hall_tail["Current (mA)"], hall_tail["Hall Voltage (V)"], s=10, label="Hall Voltage (V)", color="black")
plt.legend()
plt.grid()
plt.title("Inward B field", fontsize=16)
plt.xlabel("Current (mA)", fontsize=14)
plt.ylabel("Voltage (V)", fontsize=14)


def linearFit(x, k, b):
    return k*x+b

import scipy.optimize as optm

headFit = optm.curve_fit(linearFit, hall_head["Current (mA)"], hall_head["Voltage (V)"])
headFit_H = optm.curve_fit(linearFit, hall_head["Current (mA)"], hall_head["Hall Voltage (V)"])

print("Outward magnetic field:\n\tVoltage to Current relation: V = %f.3 A + %f.3" % (headFit[0][0], headFit[0][1]))
print("\t", r"k = %f.3 +- %f.3" %(headFit[0][0], np.sqrt(headFit[1][0,0])))
print("\t", r"b = %f.3 +- %f.3" %(headFit[0][1], np.sqrt(headFit[1][1,1])))

print("\nOutward magnetic field:\n\tHall voltage to Current relation: V = %f.3 I + %f.3" % (headFit_H[0][0], headFit_H[0][1]))
print("\t", "k = %f.3 +- %f.3" %(headFit_H[0][0], np.sqrt(headFit_H[1][0,0])))
print("\t", "b = %f.3 +- %f.3" %(headFit_H[0][1], np.sqrt(headFit_H[1][1,1])))

tailFit = optm.curve_fit(linearFit, hall_tail["Current (mA)"], hall_tail["Voltage (V)"])
tailFit_H = optm.curve_fit(linearFit, hall_tail["Current (mA)"], hall_tail["Hall Voltage (V)"])

print("Inward magnetic field:\n\tVoltage to Current relation: V = %f.3 I + %f.3" % (tailFit[0][0], tailFit[0][1]))
print("\t", r"k = %f.3 $\pm$ %f.3" %(tailFit[0][0], np.sqrt(tailFit[1][0,0])))
print("\t", r"b = %f.3 $\pm$ %f.3" %(tailFit[0][1], np.sqrt(tailFit[1][1,1])))

print("\nInward magnetic field:\n\tHall voltage to Current relation: V = %f.3 I + %f.3" % (tailFit_H[0][0], tailFit_H[0][1]))
print("\t", "k = %f.3 +- %f.3" %(tailFit_H[0][0], np.sqrt(tailFit_H[1][0,0])))
print("\t", "b = %f.3 +- %f.3" %(tailFit_H[0][1], np.sqrt(tailFit_H[1][1,1])))

plt.figure(figsize=(5,3))
plt.scatter(hall_head["Current (mA)"], hall_head["Voltage (V)"], s=35, label="Voltage data (V)", marker="3", color="orange")
plt.scatter(hall_head["Current (mA)"], hall_head["Hall Voltage (V)"], s=10, label="Hall Voltage data (V)", color="black")
plt.plot(hall_head["Current (mA)"], linearFit(hall_head["Current (mA)"], headFit[0][0], headFit[0][1]), label="Voltage fit (V)", marker="3", color="orange")
plt.plot(hall_head["Current (mA)"], linearFit(hall_head["Current (mA)"], headFit_H[0][0], headFit_H[0][1]), label="Hall Voltage fit (V)", color="black")
plt.legend()
plt.grid()
plt.title("Outward B field", fontsize=16)
plt.xlabel("Current (mA)", fontsize=14)
plt.ylabel("Voltage (V)", fontsize=14)

plt.figure(figsize=(5,3))
plt.scatter(hall_tail["Current (mA)"], hall_tail["Voltage (V)"], s=35, label="Voltage data (V)", marker="3", color="orange")
plt.scatter(hall_tail["Current (mA)"], hall_tail["Hall Voltage (V)"], s=10, label="Hall Voltage data (V)", color="black")
plt.plot(hall_tail["Current (mA)"], linearFit(hall_tail["Current (mA)"], tailFit[0][0], tailFit[0][1]), label="Voltage fit (V)", marker="3", color="orange")
plt.plot(hall_tail["Current (mA)"], linearFit(hall_tail["Current (mA)"], tailFit_H[0][0], tailFit_H[0][1]), label="Hall Voltage fit (V)", color="black")
plt.legend()
plt.grid()
plt.title("Inward B field", fontsize=16)
plt.xlabel("Current (mA)", fontsize=14)
plt.ylabel("Voltage (V)", fontsize=14)

xdata = hall_head["Current (mA)"]
ydata = hall_head["Hall Voltage (V)"]
residuals = ydata - linearFit(xdata, headFit_H[0][0], headFit_H[0][1])
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)
r_squared = 1 - (ss_res / ss_tot)
print(r_squared)


J = (60*(10**(-3)))/(10**(-5))
E = temperature["Voltage (V)"]/(10*10**(-3))
Cond = J/E

plt.figure(figsize=(5,3))
plt.scatter(temperature["Current (mA)"] + 273.15, Cond, s=35, label="Conductivity (V)", marker="3", color="orange")
plt.legend()
plt.grid()
plt.title("Figure 4", fontsize=16)
plt.xlabel("Temperature (K)", fontsize=14)
plt.ylabel("Conductivity", fontsize=14)


plt.figure(figsize=(5,3))
plt.scatter(temperature["Current (mA)"] + 273.15, temperature['Hall Voltage (V)'], s=35, label="Hall Voltage (V)", marker="3", color="purple")
plt.legend()
plt.grid()
plt.title("Figure 3", fontsize=16)
plt.xlabel("Temperature (K)", fontsize=14)
plt.ylabel("Hall Voltage (V)", fontsize=14)

e = 1.60217663*10**(-19)
k = 1.3807*10**(-23)
def sigma(t, a, b, c, Eg):
    res = a + b/t + c*np.exp(-e*Eg /(2*k*t))
    return res

tempFit = optm.curve_fit(sigma, temperature["Current (mA)"] + 273.15, Cond, p0=(-15, 10e3, 6e6, 1.0))
tempFit[0][3], np.sqrt(tempFit[1][3,3])

T_arr = np.linspace(30,142, 50) + 273.15

plt.figure(figsize=(5,3))
plt.scatter(temperature["Current (mA)"] + 273.15, Cond, s=55, label="Conductivity", marker="+", color="darkorange")
plt.plot(T_arr , sigma(T_arr, tempFit[0][0], tempFit[0][1], tempFit[0][2], tempFit[0][3]),
         label="Best Fit", color="black", ls="--", alpha=0.6)
plt.legend()
plt.grid()
plt.title("Figure 5", fontsize=16)
plt.xlabel("Temperature (K)", fontsize=14)
plt.ylabel("Conductivity", fontsize=14)
