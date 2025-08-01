import numpy as np
import random
import matplotlib.pyplot as plt

d = 100
D = 1
a = 50
N = 10
Xs = np.linspace(-a/2, a/2, N)
r1s = [np.sqrt(D**2 + (Xs[i]+d/2)**2) for i in range(N)]
r2s = [np.sqrt(D**2 + (Xs[i]-d/2)**2) for i in range(N)]
sum = 0
w=1

phase = 0
for trial in range(1000):
    Phis = [random.random()*2*np.pi for m in range(N)]
    s = 0
    p = 0
    for k in range(N):
        for j in range(N):
            s += np.exp(1j*w*(r1s[k]-r1s[j]) - 1j*(Phis[j]-Phis[k]))
            p += np.exp(-1j*(Phis[j]-Phis[k]))
    phase += p
    sum += s

average = sum/1000
print(average)

d = 100
D = 1
a = 50
N = 15
Xs = np.linspace(-a/2, a/2, N)
r1s = [np.sqrt(D**2 + (Xs[i]+d/2)**2) for i in range(N)]
r2s = [np.sqrt(D**2 + (Xs[i]-d/2)**2) for i in range(N)]
sum = 0
w=1
for j in range(N):
    for k in range(N):
        s = np.exp(1j*w*(r1s[k]-r1s[j]))
        sum += s

print(sum)


d = 2
D = 1
a = 1
N = 3
Xs = np.linspace(-a/2, a/2, N)
r1s = [np.sqrt(D**2 + (Xs[i]+d/2)**2) for i in range(N)]
r2s = [np.sqrt(D**2 + (Xs[i]-d/2)**2) for i in range(N)]
sum = 0
w=1

for trial in range(10000):
    s = 0
    Phi1s = [random.random()*2*np.pi for m in range(N)]
    Phi2s = [random.random()*2*np.pi for m in range(N)]
    for k in range(N):
        for j in range(N):
            for jp in range(N):
                for kp in range(N):
                    s += np.exp(-1j*w*(r1s[k]+r2s[kp]-r1s[j]-r2s[jp])
                                +1j*(Phi1s[j]+Phi2s[jp]-Phi1s[k]-Phi2s[kp]))
    sum += s

average = sum/10000
print(average)
