import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def QandW(ar, ai, br, bi):
    a_mag2 = ar**2 + ai**2
    b_mag2 = br**2 + bi**2
    aminusb_mag2 = (ar-br)**2 + (ai-bi)**2
    aplusb_mag2 = (ar+br)**2 + (ai+bi)**2
    b_ac = (br+1j*bi)*(ar-1j*ai)
    a_bc = (ar+1j*ai)*(br-1j*bi)
    W_term1 = np.exp(-2*aminusb_mag2)
    W_term2 = np.exp(-2*aplusb_mag2)
    W_term3 = 2*np.exp(-2*a_mag2)*np.cos(2*np.abs(b_ac-a_bc))
    W = (1/np.pi)*(W_term1+W_term2+W_term3)
    Q_coeff = (1/(2*np.pi))*np.exp(-a_mag2-b_mag2)
    Q = Q_coeff*np.abs(np.exp(a_bc)+np.exp(-a_bc))**2
    return Q, W

linspace = np.linspace(-6, 6, 1000)
X, Y = np.meshgrid(linspace, linspace)
Qs, Ws = QandW(X, Y, 1.5, 1.5)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(X, Y, Qs, cmap='seismic',
                       linewidth=0, antialiased=False)
ax.set_zlim(-0.03, 0.15)
ax.zaxis.set_major_locator(LinearLocator(5))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=10)
ax.set_xlabel(r'Re($\alpha$)', fontsize=12)
ax.set_ylabel(r'Im($\alpha$)', fontsize=12)
ax.set_zlabel(r'Q($\alpha$)', fontsize=12)
plt.show()


linspace = np.linspace(-2.5, 2.5, 1000)
X, Y = np.meshgrid(linspace, linspace)
Qs, Ws = QandW(X, Y, 1.5, 1.5)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(X, Y, Ws, cmap='seismic',
                       linewidth=0, antialiased=False)
ax.set_zlim(-0.6, 0.6)
ax.zaxis.set_major_locator(LinearLocator(5))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=10)
ax.set_xlabel(r'Re($\alpha$)', fontsize=12)
ax.set_ylabel(r'Im($\alpha$)', fontsize=12)
ax.set_zlabel(r'W($\alpha$)', fontsize=12)
plt.show()
