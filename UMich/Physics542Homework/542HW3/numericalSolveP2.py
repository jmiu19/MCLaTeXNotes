import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# discretize time
delta_t = 0.00001
ts = np.linspace(0,15,int(20/delta_t))

# parameter settings
gamma = 1/2
gamma2 = 2*gamma
delta = 0.1
Omega0s = [0.2, 3]

# initial conditions
u_i = 0 # tilde_rho12 + tilde_rho21
v_i = 0 # 1j*(tilde_rho21 - tilde_rho12)
w_i = -1 # rho22 - rho11
m_i = 1 # rho22 + rho11

## verification
rho11_i = 1
rho22_i = 0
tilde_rho12_i = 0
tilde_rho21_i = 0

colors = ['red', 'blue']

for i in range(len(Omega0s)):
    Omega0 = Omega0s[i]
    chi = Omega0/2
    rho11_array = [rho11_i]
    rho22_array = [rho22_i]
    tilde_rho12_array = [tilde_rho12_i]
    tilde_rho21_array = [tilde_rho21_i]
    for t in ts[:-1]:
        rho11, rho22 = rho11_array[-1], rho22_array[-1]
        tilde_rho12, tilde_rho21 = tilde_rho12_array[-1], tilde_rho21_array[-1]
        dot_rho11 = (-1j*np.abs(chi)*tilde_rho21
                        + 1j*np.abs(chi)*tilde_rho12
                            + gamma2*rho22)
        dot_rho22 = (1j*np.abs(chi)*tilde_rho21
                        - 1j*np.abs(chi)*tilde_rho12
                            - gamma2*rho22)
        dot_tilde_rho12 = (1j*delta*tilde_rho12
                                - 1j*np.abs(chi)*(rho22-rho11)
                                    - gamma*tilde_rho12)
        dot_tilde_rho21 = (-1j*delta*tilde_rho21
                                + 1j*np.abs(chi)*(rho22-rho11)
                                    -gamma*tilde_rho21)
        rho11_array.append(rho11_array[-1]+delta_t*dot_rho11)
        rho22_array.append(rho22_array[-1]+delta_t*dot_rho22)
        tilde_rho12_array.append(tilde_rho12_array[-1]+delta_t*dot_tilde_rho12)
        tilde_rho21_array.append(tilde_rho21_array[-1]+delta_t*dot_tilde_rho21)
    plt.plot(ts, np.real([rho22_array[i]-rho11_array[i]
                                for i in range(len(rho22_array))]),
             linestyle='--', color=colors[i])


for i in range(len(Omega0s)):
    Omega0 = Omega0s[i]
    u_array = [u_i]
    v_array = [v_i]
    w_array = [w_i]
    for t in ts[:-1]:
        u, v, w = u_array[-1], v_array[-1], w_array[-1]
        dot_u = -delta*v - gamma*u
        dot_v = delta*u - np.abs(Omega0)*w - gamma*v
        dot_w = np.abs(Omega0)*v - gamma2*(w+1)
        u_array.append(u_array[-1]+delta_t*dot_u)
        v_array.append(v_array[-1]+delta_t*dot_v)
        w_array.append(w_array[-1]+delta_t*dot_w)
    plt.plot(ts, np.real(w_array), color=colors[i], alpha=0.5,
             label=r'w ($\Omega_0$='+str(Omega0)+')')


plt.ylabel(r'w', fontsize='xx-large')
plt.xlabel("t",fontsize='xx-large')
plt.legend(fontsize='xx-large')
plt.tight_layout()
plt.show()
