import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


Ns = range(50)
alpha = np.sqrt(30)
lamb = 1

def S(alpha, t):
    SumSum = 0
    for n in Ns:
        for m in Ns:
            coeff = np.exp(-2*(np.abs(alpha)**2))*(np.abs(alpha)**(2*(n+m)))/(np.math.factorial(n)*np.math.factorial(m))
            sinCoeff = np.sqrt(n*m)/(np.abs(alpha)**2)
            sin = np.sin(lamb*t*np.sqrt(n))*np.sin(lamb*t*np.sqrt(m))
            cos = np.cos(lamb*t*np.sqrt(n+1))*np.cos(lamb*t*np.sqrt(m+1))
            SumSum += coeff*(np.abs(sinCoeff*sin+cos)**2)
    return 1-SumSum


def Q(alpha, beta, t):
    coeff = np.exp(-np.abs(beta)**2-np.abs(alpha)**2)/np.pi
    SumSum = 0
    for n in Ns:
        for m in Ns:
            coeffSumSum = ((alpha*np.conjugate(beta))**n)*((beta*np.conjugate(alpha))**m)/(np.math.factorial(n)*np.math.factorial(m))
            sinCoeff = np.sqrt(n*m)/(np.abs(alpha)**2)
            sin = np.sin(lamb*t*np.sqrt(n))*np.sin(lamb*t*np.sqrt(m))
            cos = np.cos(lamb*t*np.sqrt(n+1))*np.cos(lamb*t*np.sqrt(m+1))
            SumSum += coeffSumSum*(sinCoeff*sin+cos)
    return coeff*SumSum


ts = np.linspace(0,40,500)
plt.plot(ts, [S(alpha, t) for t in ts], label=r'$S(t)$')
plt.ylabel(r'$S(t)$', fontsize='xx-large')
plt.xlabel("t",fontsize='xx-large')
plt.grid()
plt.tight_layout()
plt.show()


ts = [0, 5, 10, 15, 20, 25, 30, 35, 40]
for t in ts:
    linspace = np.linspace(-8, 8, 65)
    X, Y = np.meshgrid(linspace, linspace)
    Qs = [[Q(alpha, X[j][i]+1j*Y[j][i], t) for i in range(65)] for j in range(65)]
    Qs = np.array(Qs)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Qs, cmap='seismic',
                           linewidth=0, antialiased=False)
    ax.set_zlim(-0.03, 0.5)
    ax.zaxis.set_major_locator(LinearLocator(5))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=10)
    ax.set_xlabel(r'Re($\beta$)', fontsize=12)
    ax.set_ylabel(r'Im($\beta$)', fontsize=12)
    ax.set_zlabel(r'Q($\beta$)', fontsize=12)
    plt.savefig('Q('+str(t)+').png')
