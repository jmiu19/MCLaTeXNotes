import numpy as np
import random
import matplotlib.pyplot as plt

omega = 1
ts = np.linspace(0,1,10)
S_averages = []
S_sqrd_averages = []

for k in range(100000):
    # a single realization over time
    phases = [random.random()*2*np.pi for i in range(100)]
    sum_t = 0
    S_ts = []
    for t in ts:
        sum_n = 0
        for i in range(100):
            phase = phases[i]
            sum_n += np.exp(-1j*omega*t + 1j*phase)
        S_ts.append((np.abs(sum_n))**2)
        sum_t += S_ts[-1]
    # compute average of S over time
    S_timeAveraged = sum_t / len(ts)
    S_averages.append(S_timeAveraged)
    S_sqrd_averages.append(S_timeAveraged**2)
    if (k<=2):
        print(r'$\langle S\rangle_t = $'
                          +str(round(S_timeAveraged,2)))


S_phaseTimeAveraged = np.sum(S_averages)/100000
S_sqrd_phaseTimeAveraged = np.sum(S_sqrd_averages)/100000

plt.hist(S_averages, bins=50)
plt.axvline(x=S_phaseTimeAveraged, color='red')
plt.annotate(r'$\langle S\rangle =$'+str(round(S_phaseTimeAveraged,2)),
             (S_phaseTimeAveraged+5,80))
plt.ylabel(r'realizations in 100000 trials', fontsize='x-large')
plt.xlabel("S averaged over time",fontsize='x-large')
plt.show()


plt.hist(S_sqrd_averages, bins=50)
plt.axvline(x=S_sqrd_phaseTimeAveraged, color='red')
plt.annotate(r'$\langle S^2\rangle =$'+str(round(S_sqrd_phaseTimeAveraged,2)),
             (S_sqrd_phaseTimeAveraged+5000,300))
plt.ylabel(r'realizations in 100000 trials', fontsize='x-large')
plt.xlabel(r"$S^2$ averaged over time",fontsize='x-large')
plt.show()

print(S_phaseTimeAveraged, 100)
print(S_sqrd_phaseTimeAveraged, 2*(100**2-100))
