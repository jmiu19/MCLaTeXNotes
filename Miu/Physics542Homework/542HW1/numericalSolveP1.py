import numpy as np
import matplotlib.pyplot as plt

delta_t = 0.00005
ts = np.linspace(0,2,int(2/delta_t))

omega = 0
Omega0s = [0.5,1,2,10]
omega0 = 5

red_colors = [(1,0,0), (1,0,0), (1,0,0), (1,0,0)]
blue_colors = [(0,0,1), (0,0,1), (0,0,1), (0,0,1)]

def a1_sqamp_ana(t, omega0, a1i, a2i, Omega0):
    # squared amplitude of a1 analytic function
    omega0 = -omega0
    y = 2*Omega0
    X = ((omega0**2)+(y**2))**(1/2)
    a1 = (np.cos(X*t/2)+1j*omega0/X*np.sin(X*t/2))*a1i-(1j*y/X)*np.sin(X*t/2)*a2i
    return a1*np.conjugate(a1)

def a2_sqamp_ana(t, omega0, a1i, a2i, Omega0):
    # squared amplitude of a2 analytic function
    omega0 = -omega0
    y = 2*Omega0
    X = ((omega0**2)+(y**2))**(1/2)
    a2 = (np.cos(X*t/2)-1j*omega0/X*np.sin(X*t/2))*a2i-(1j*y/X)*np.sin(X*t/2)*a1i
    return a2*np.conjugate(a2)

# def a1_sqamp_ana(t, delta, a1i, a2i, Omega0):
#     #squared amplitude of a1 analytic function
#     theta = Omega0*np.sin(omega*t)/omega
#     a1 = np.cos(theta)*c1i - 1j*np.sin(theta)*c2i
#     return a1*np.conjugate(a1)
#
# def a2_sqamp_ana(t, delta, a1i, a2i, Omega0):
#     #squared amplitude of a1 analytic function
#     theta = Omega0*np.sin(omega*t)/omega
#     a2 = -1j*np.sin(theta)*c1i + np.cos(theta)*c2i
#     return a2*np.conjugate(a2)


# numerical computation
for Omega0 in Omega0s:
    delta = omega0-omega
    c1i = 1
    c2i = 0
    a1 = [c1i]
    a2 = [c2i]
    hbar = 1
    m = 1j*omega0*hbar/2
    c = -1j*Omega0

    for t in ts[:-1]:
        dot_a1 = m*a1[-1] + c*np.cos(omega*t)*a2[-1]
        dot_a2 = c*np.cos(omega*t)*a1[-1] - m*a2[-1]
        a1.append(a1[-1]+delta_t*dot_a1)
        a2.append(a2[-1]+delta_t*dot_a2)

    a1_sqamp = np.real(np.array([a*np.conjugate(a) for a in a1]))
    a2_sqamp = np.real(np.array([a*np.conjugate(a) for a in a2]))

    # general plots
    plt.plot(ts, a1_sqamp, color=red_colors[n], label='level 1 numerical')
    plt.plot(ts, a2_sqamp, color=blue_colors[n], label='level 2 numerical')
    plt.plot(ts, [a1_sqamp_ana(t, omega0, c1i, c2i, Omega0) for t in ts],
             label='level 1 analytic', linestyle='dashed', color='orange')
    plt.plot(ts, [a2_sqamp_ana(t, omega0, c1i, c2i, Omega0) for t in ts],
             label='level 2 analytic', linestyle='dashed', color='purple')
    plt.ylabel(r'amplitude', fontsize='xx-large')
    plt.xlabel("t/T",fontsize='xx-large')
    plt.legend(fontsize='xx-large')
    plt.title(r'Omega_0='+str(Omega0), fontsize='xx-large')
    plt.tight_layout()
    plt.show()
