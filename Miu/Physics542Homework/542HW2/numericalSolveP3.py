import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def normalize(wf):
    normalization = np.sqrt(wf[0]**2 + wf[1]**2)
    return wf/normalization

def evolve_wavefunction_dt(wf_init, wf, Gamma, t_old, t_new):
    alpha_0 = wf_init[0]
    beta_0 = wf_init[1]
    alpha_old = wf[0]
    beta_old = wf[1]
    if beta_old == 0:
        return wf
    else:
        dp = Gamma*(abs(beta_old)**2)*(t_new-t_old)
        epsilon = np.random.random()
        if dp > epsilon:
            wf = np.array([1,0])
            return wf
        else:
            denominator = np.sqrt(abs(alpha_0)**2+abs(beta_0)**2*np.exp(-Gamma*t_new))
            alpha_new = alpha_0/denominator
            beta_new = beta_0*np.exp(-Gamma*t_new/2)/denominator
            wf = normalize(np.array([alpha_new,beta_new]))
            return wf

def evolve_wavefunction_finite(t_arr, Gamma, Psi_init):
    wf_evolved = [Psi_init]
    for i in range(len(t_arr)-1):
        t_old = t_arr[i]+t_arr[0]
        t_new = t_arr[i+1]+t_arr[0]
        wf_old = wf_evolved[-1]
        wf_new = evolve_wavefunction_dt(Psi_init, wf_old, Gamma, t_old, t_new)
        wf_evolved.append(wf_new)
    return wf_evolved

def average_over_realizations(t_arr, Gamma, N, Psi_init):
    N_wf_sq_array = []
    for n in range(N):
        wf_evolved = evolve_wavefunction_finite(t_arr, Gamma, Psi_init)
        wf_evolved_sq = np.array(wf_evolved)**2
        N_wf_sq_array.append(wf_evolved_sq)
    wf_sq_averaged = np.mean(np.array(N_wf_sq_array), axis=0)
    return wf_sq_averaged

def exponential(t, Gamma, amp):
    return np.exp(-t*Gamma)*amp


Gamma = 5 # decay rate
dt = 0.001 # the time jump
t = Gamma/5. # the final time
t_arr = np.linspace(0,t,int(t/dt)) # the values of t in the time interval
N = 500 # number of realizations to average over

# first wavefunction
Unnormalized_Psi_init1 = np.array([0,1])
Unnormalized_Psi_init2 = np.array([1,1])
Psi_init1 = normalize(Unnormalized_Psi_init1)
Psi_init2 = normalize(Unnormalized_Psi_init2)

wf_sq_averaged1 = average_over_realizations(t_arr, Gamma, N, Psi_init1)
wf_sq_averaged2 = average_over_realizations(t_arr, Gamma, N, Psi_init2)

## averaged wavefunction plot
alphas_sqd1 = [np.abs(np.array(wf[0])) for wf in wf_sq_averaged1]
betas_sqd1 = [np.abs(np.array(wf[1])) for wf in wf_sq_averaged1]
alphas_sqd2 = [np.abs(np.array(wf[0])) for wf in wf_sq_averaged2]
betas_sqd2 = [np.abs(np.array(wf[1])) for wf in wf_sq_averaged2]
plt.plot(t_arr, alphas_sqd1, color='r', label='level 1 (I)', alpha=0.5)
plt.plot(t_arr, betas_sqd1, color='b', label='level 2 (I)', alpha=0.5)
plt.plot(t_arr, exponential(t_arr, Gamma, 1), color='black', alpha=0.3, linestyle='--')
plt.ylabel(r'sqrd amp', fontsize='xx-large')
plt.xlabel("t",fontsize='xx-large')
plt.legend(fontsize='xx-large')
plt.tight_layout()
plt.savefig('I_averaged.png')
plt.clf()

plt.plot(t_arr, alphas_sqd2, color='r', label='level 1 (II)', alpha=0.5)
plt.plot(t_arr, betas_sqd2, color='b', label='level 2 (II)', alpha=0.5)
plt.ylabel(r'sqrd amp', fontsize='xx-large')
plt.plot(t_arr, exponential(t_arr, Gamma, 0.5), color='black', alpha=0.3, linestyle='--')
plt.xlabel("t",fontsize='xx-large')
plt.legend(fontsize='xx-large')
plt.tight_layout()
plt.savefig('II_averaged.png')
plt.clf()


for i in [1,2,3,4,5]:
    wf_single1 = evolve_wavefunction_finite(t_arr, Gamma, Psi_init1)
    wf_single2 = evolve_wavefunction_finite(t_arr, Gamma, Psi_init2)
    ## single wavefunction plot (initial data 1)
    alphas_sqd1 = [np.abs(np.array(wf[0]))**2 for wf in wf_single1]
    betas_sqd1 = [np.abs(np.array(wf[1]))**2 for wf in wf_single1]
    plt.plot(t_arr, alphas_sqd1, color='r', label='level 1 (I)')
    plt.plot(t_arr, betas_sqd1, color='b', label='level 2 (I)')
    plt.ylabel(r'sqrd amp', fontsize='xx-large')
    plt.xlabel("t",fontsize='xx-large')
    plt.legend(fontsize='xx-large')
    plt.tight_layout()
    plt.savefig('I/'+str(i)+'I_single.png')
    plt.clf()
    ## single wavefunction plot (initial data 2)
    alphas_sqd2 = [np.abs(np.array(wf[0]))**2 for wf in wf_single2]
    betas_sqd2 = [np.abs(np.array(wf[1]))**2 for wf in wf_single2]
    plt.plot(t_arr, alphas_sqd2, color='r', linestyle='--', label='level 1 (II)')
    plt.plot(t_arr, betas_sqd2, color='b', linestyle='--', label='level 2 (II)')
    plt.ylabel(r'sqrd amp', fontsize='xx-large')
    plt.xlabel("t",fontsize='xx-large')
    plt.legend(fontsize='xx-large')
    plt.tight_layout()
    plt.savefig('II/'+str(i)+'II_single.png')
    plt.clf()


Unnormalized_Psi_ini0 = np.array([1,0])
Psi_init0 = normalize(Unnormalized_Psi_ini0)
wf_single0 = evolve_wavefunction_finite(t_arr, Gamma, Psi_init0)
## single wavefunction plot (initial data 1)
alphas_sqd1 = [np.abs(np.array(wf[0]))**2 for wf in wf_single0]
betas_sqd1 = [np.abs(np.array(wf[1]))**2 for wf in wf_single0]
plt.plot(t_arr, alphas_sqd1, color='r', label='level 1 (I)')
plt.plot(t_arr, betas_sqd1, color='b', label='level 2 (I)')
plt.ylabel(r'sqrd amp', fontsize='xx-large')
plt.xlabel("t",fontsize='xx-large')
plt.legend(fontsize='xx-large')
plt.tight_layout()
plt.savefig('III_single.png')
plt.clf()
