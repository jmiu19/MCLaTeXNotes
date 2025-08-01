import numpy as np
import matplotlib.pyplot as plt

## discretize for numerical computation
delta_t = 0.00001
ts = np.linspace(-10,10,int(20/delta_t))
## discretize for analytic plot
delta_t_ana = 0.005
ts_ana = np.linspace(-10,10,int(20/delta_t_ana))

# initial conditions
c1i = 1
c2i = 0
c1 = [c1i]
c2 = [c2i]
# parameters
hbar = 1
Omega0Mod = 30
delta0 = 30

def Omega0Func(t):
    return Omega0Mod*np.exp(-(t**2))

def deltaFunc(t):
    if t<0:
        delta = delta0*((1-np.exp(t))**3)
    else:
        delta = 0
    return delta

def c1_dressed_ana(ts_truncated,ini0):
    ## dressed c1 analytic form
    Omegas = []
    for t in ts_truncated:
        Omega0 = Omega0Func(t)
        delta = deltaFunc(t)
        Omega = np.sqrt(delta**2 + np.absolute(Omega0)**2)
        Omegas.append(Omega)
    I = np.trapz(Omegas, x=ts_truncated)
    return np.exp(1j*I/2)*ini0

def c2_dressed_ana(ts_truncated,ini0):
    ## dressed c2 analytic form
    Omegas = []
    for t in ts_truncated:
        Omega0 = Omega0Func(t)
        delta = deltaFunc(t)
        Omega = np.sqrt(delta**2 + np.absolute(Omega0)**2)
        Omegas.append(Omega)
    I = np.trapz(Omegas, x=ts_truncated)
    return np.exp(-1j*I/2)*ini0

def c_tilde_ana(t, c1_dressed, c2_dressed):
    ## convert from dressed c to tilde c
    Omega0 = Omega0Func(t)
    delta = deltaFunc(t)
    Omega = np.sqrt(delta**2 + np.absolute(Omega0)**2)
    sin = np.sqrt((1/2)*(1-delta/Omega))
    cos = np.sqrt((1/2)*(1+delta/Omega))
    c1_tilde_ana = cos*c1_dressed+sin*c2_dressed
    c2_tilde_ana = -sin*c1_dressed+cos*c2_dressed
    return c1_tilde_ana, c2_tilde_ana


## numerical method computed as follows
for t in ts[:-1]:
    Omega0 = Omega0Func(t)
    delta = deltaFunc(t)
    c = -1j*np.conjugate(Omega0)*np.exp(-1j*delta*t)/2
    cc = -1j*Omega0*np.exp(1j*delta*t)/2
    dot_c1 = c*c2[-1]
    dot_c2 = cc*c1[-1]
    c1.append(c1[-1]+delta_t*dot_c1)
    c2.append(c2[-1]+delta_t*dot_c2)

c1 = np.array(c1)
c2 = np.array(c2)
rho12 = np.absolute(c1*np.conjugate(c2))

## analytic results computed as follows
# initial condition for dressed c
t_ini = ts_ana[0]
Omega0_ini = Omega0Func(t_ini)
delta_ini = deltaFunc(t_ini)
Omega_ini = np.sqrt(delta_ini**2 + np.absolute(Omega0_ini)**2)
sin_ini = np.sqrt((1/2)*(1-delta_ini/Omega_ini))
cos_ini = np.sqrt((1/2)*(1+delta_ini/Omega_ini))
c1_tilde_ini = np.exp(1j*delta_ini*t_ini/2)*c1i
c2_tilde_ini = -np.exp(1j*delta_ini*t_ini/2)*c2i
c1_dressed_ini = cos_ini*c1_tilde_ini-sin_ini*c2_tilde_ini
c2_dressed_ini = sin_ini*c1_tilde_ini+cos_ini*c2_tilde_ini

c1_tilde_anas = [c1_tilde_ini]
c2_tilde_anas = [c2_tilde_ini]

# compute the analytic values
for i in range(len(ts_ana)):
    t = ts_ana[i]
    c1_dressed = c1_dressed_ana(ts_ana[:i],c1_dressed_ini)
    c2_dressed = c2_dressed_ana(ts_ana[:i],c2_dressed_ini)
    c1_tilde, c2_tilde = c_tilde_ana(t, c1_dressed, c2_dressed)
    c1_tilde_anas.append(c1_tilde)
    c2_tilde_anas.append(c2_tilde)

# compute analytic magnitude
c1_tilde_mag = np.absolute(np.array(c1_tilde_anas))
c2_tilde_mag = np.absolute(np.array(c2_tilde_anas))
rho12_adbatic = np.absolute(c1_tilde_anas*np.conjugate(c2_tilde_anas))

## truncate t interval for plotting
length = len(ts)
L31 = int(length/3)
L32 = 2*int(length/3)

length = len(ts_ana)
L31ana = int(length/3)
L32ana = 2*int(length/3)

## plot the results
plt.plot(ts[L31:L32], rho12[L31:L32],
         label=r'$\rho_{12}(\tau)$', color='black')
plt.plot(ts_ana[L31ana:L32ana], rho12_adbatic[L31ana:L32ana],
         label=r'$\rho_{12}(\tau)$ adbatic', linestyle='--', color='gray')
plt.plot(ts[L31:L32], np.absolute(c1)[L31:L32],
         label=r'$c_1(\tau)$', alpha=0.3, color='red')
plt.plot(ts[L31:L32], np.absolute(c2)[L31:L32],
         label=r'$c_2(\tau)$', alpha=0.3, color='blue')
plt.plot(ts_ana[L31ana:L32ana], c1_tilde_mag[L31ana:L32ana],
         label=r'$c_1(\tau)$ adbatic', linestyle='--',
         alpha=0.3, color='orange')
plt.plot(ts_ana[L31ana:L32ana], c2_tilde_mag[L31ana:L32ana],
         label=r'$c_2(\tau)$ adbatic', linestyle='--',
         alpha=0.3, color='purple')
plt.plot(ts[L31:L32], ([0.5]*len(ts))[L31:L32],
         linestyle='--', alpha=0.3, color='black')
plt.annotate('amp. = 0.5', (-3,0.45), alpha=0.3)
plt.ylabel(r'amplitude', fontsize='xx-large')
plt.xlabel("t/T",fontsize='xx-large')
plt.legend(fontsize='x-large')
plt.tight_layout()
plt.show()
