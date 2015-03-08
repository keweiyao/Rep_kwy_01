import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def V(x, y):
    return 10.0*(x>-2)*(x<2)*(y<2)*(y>-2)

def update_V(dt, phi, X, Y):
    return np.exp(-1j*dt*V(X, Y))*phi

def update_T(dt, phi, Kx, Ky):
    N = np.shape(Kx)[0]
    
    psi = np.fft.fft2(phi)
    for i in range(0, N):
        for j in range(0, N):
            psi[i][j] = np.exp(-1j*dt*(Kx[(i+int(0.5*N))%N][(j+int(0.5*N))%N]**2+Ky[(i+int(0.5*N))%N][(j+int(0.5*N))%N]**2))*psi[i][j]
    phi = np.fft.ifft2(psi)
    return phi


def solver(phi, X, Y, t_m):
    #phi_in = np.copy(phi)
    N_x = np.shape(X)[0]
    dx = x[1]-x[0]
    dt = 0.1
    kx = np.linspace(-2,2,N_x)
    ky = np.linspace(-2,2,N_x)
    Kx, Ky = np.meshgrid(kx, ky)
    
    ax = plt.gca(projection='3d')
    ax.plot_surface(X, Y, phi, rstride=1, cstride=1, cmap=cm.YlGnBu_r)
    
    phi = update_V(dt/2., phi, X, Y)
    phi = update_T(dt, phi, Kx, Ky)
    #phi_in = update_T(dt, phi_in, Kx, Ky)
    
    N_max = int(t_m/dt)
    for i in range(N_max):
        phi = update_V(dt, phi, X, Y)
        phi = update_T(dt, phi, Kx, Ky)
        #phi_in = update_T(dt, phi_in, Kx, Ky)
        print i
        if i%20==0:
            plt.clf()
            #ax = plt.gca(projection='3d')
            #ax.plot_surface(X, Y, np.abs(phi), rstride=1, cstride=1, cmap=cm.YlGnBu_r, linewidth = 0)
            plt.contour(X, Y, np.abs(phi),50)
            plt.plot([-2,-2,2,2],[2,-2,2,-2],'ro')
            plt.savefig("%d.jpeg"%i)
            #plt.pause(0.01)
    phi = update_V(dt/2., phi, X, Y)
    return phi

#____main_______
x = np.linspace(-50,50,100)
y = np.linspace(-50,50,100)
X, Y = np.meshgrid(x, y)
x0 = -15
sigma = 5
k = 1
phi0 = np.exp(-0.5*((X-x0)/sigma)**2-0.5*(Y/sigma)**2+1j*k*X)
plt.figure(figsize = (7, 7))
phi = solver(phi0, X, Y, 50.0)
plt.show()

