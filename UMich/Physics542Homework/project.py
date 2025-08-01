import numpy as np
import matplotlib.pyplot as plt
from qutip import *

pi = np.pi
omega0 = 10
omega = 9

As = np.linspace(0, 8, 1000) * omega
T = (2*pi)/omega
q_energies = np.zeros((len(As), 2))

H0 = omega0/2.0 * sigmaz()
args = {'w': omega}
for idx, A in enumerate(As):
    H1 = A * sigmax()
    H = [H0, [H1, 'cos(w * t)']]
    f_modes, f_energies = floquet_modes(H, T, args, True)
    q_energies[idx,:] = f_energies

plt.figure()
plt.plot(As/omega, q_energies[:,0] / delta, 'b', As/omega, q_energies[:,1] / delta, 'r')
plt.xlabel(r'$\Omega/\omega$', fontsize='x-large')
plt.ylabel(r'Quasienergy / $\delta$', fontsize='x-large')
plt.show()










from pylab import *
from scipy import *

delta_t = 0.00001
min_t = 0
max_t = 2
tlist  = np.linspace(min_t, max_t, 101)
ts = np.linspace(min_t, max_t, int((max_t-min_t)/delta_t))


omega0 = 10
omega  = 3
Omega  = 1*omega #1.004992*omega
T      = (2*pi)/omega
psi0   = basis(2,0)

H0 = omega0/2.0 * sigmaz()
H1 = Omega * sigmax()
args = {'w': omega}
H = [H0, [H1, 'cos(w * t)']]

# find the floquet modes for the time-dependent hamiltonian
f_modes_0,f_energies = floquet_modes(H, T, args)

# decompose the inital state in the floquet modes
f_coeff = FloquetBasis(H, T, args).to_floquet_basis(psi0)

# calculate the wavefunctions using the from the floquet modes
p_ex = zeros(len(tlist))
for n, t in enumerate(tlist):
    psi_t = floquet_wavefunction_t(f_modes_0, f_energies, f_coeff, t, H, T, args)
    p_ex[n] = expect(num(2), psi_t)






delta = omega0-omega
c1i = 1
c2i = 0


a1 = [c1i]
a2 = [c2i]
m = omega0/2j
c = Omega/2j
for t in ts[:-1]:
    dot_a1 = - m*a1[-1] + c*np.exp(1j*omega*t)*a2[-1]
    dot_a2 = c*np.exp(-1j*omega*t)*a1[-1] + m*a2[-1]
    a1.append(a1[-1]+delta_t*dot_a1)
    a2.append(a2[-1]+delta_t*dot_a2)

a1_sqamp = np.real(np.array([a*np.conjugate(a) for a in a1]))
a2_sqamp = np.real(np.array([a*np.conjugate(a) for a in a2]))

a1_woRWA = [c1i]
a2_woRWA = [c2i]
m = omega0/2j
c = Omega/1j
for t in ts[:-1]:
    dot_a1 = - m*a1_woRWA[-1] + c*np.cos(omega*t)*a2_woRWA[-1]
    dot_a2 = c*np.cos(omega*t)*a1_woRWA[-1] + m*a2_woRWA[-1]
    a1_woRWA.append(a1_woRWA[-1]+delta_t*dot_a1)
    a2_woRWA.append(a2_woRWA[-1]+delta_t*dot_a2)

a1_sqamp_woRWA = np.real(np.array([a*np.conjugate(a) for a in a1_woRWA]))
a2_sqamp_woRWA = np.real(np.array([a*np.conjugate(a) for a in a2_woRWA]))



# general plots
plot(tlist, real(p_ex),   'b--', label=r"Floquet $P_1$")
plot(tlist, 1-real(p_ex), 'r--', label=r"Floquet $P_0$")
plot(ts, a1_sqamp, 'r:', alpha=0.5, label=r'RWA $P_0$')
plot(ts, a2_sqamp, 'b:', alpha=0.5, label=r'RWA $P_1$')
plot(ts, a1_sqamp_woRWA, color='r', alpha=0.2, label='Num. $P_0$')
plot(ts, a2_sqamp_woRWA, color='b', alpha=0.2, label='Num. $P_1$')
xlabel('Time', fontsize='x-large')
ylabel('Occupation probability', fontsize='x-large')
#legend(fontsize='x-large')
tight_layout()
show()
