import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from progressbar import *

def V(x, y):
    r = np.sqrt(x**2+y**2)
    return -10/r*np.exp(-r)+1/r**2

def update_V(T_V, phi):
    phi = T_V*phi
    return np.array(phi, dtype = "complex")

def update_T(T_T, phi):
    psi = np.fft.fft2(phi)
    psi = T_T*psi
    phi = np.fft.ifft2(psi)
    return np.array(phi, dtype = "complex")


def solver(phi0, Lx, Hx, Ly, Hy, N, t_m):
    dt = 0.1
    x = np.linspace(Lx, Hx, N)
    y = np.linspace(Ly, Hy, N)
    kx = np.linspace(-2, 2, N)
    ky = np.linspace(-2, 2, N)
    
    Y, X = np.meshgrid(x,y)
    Ky, Kx = np.meshgrid(x,y)
    
    phi = np.zeros([N,N], dtype = "complex")
    Vkk_half = np.zeros([N, N], dtype = "complex")
    Vxy_half = np.zeros([N, N], dtype = "complex")
    Vkk_full = np.zeros([N, N], dtype = "complex")
    Vxy_full = np.zeros([N, N], dtype = "complex")
    
    print "Calculating half step splited operators O(V, dt/2), O(T, dt/2) and full step operators O(V, dt), O(T, dt)"
    
    for i in range(0, N):
        for j in range(0, N):
            Vkk_half[i][j] = np.exp(-1j*dt/2.*(kx[(i+int(0.5*N))%N]**2+ky[(j+int(0.5*N))%N]**2))
            Vxy_half[i][j] = np.exp(-1j*dt/2.*V(x[i], y[j]))
    Vkk_full = Vkk_half*Vkk_half
    Vxy_full = Vxy_half*Vxy_half

    ax = plt.gca(projection='3d')
    ax.plot_surface(X, Y, np.abs(phi), rstride=1, cstride=1, cmap=cm.YlGnBu_r)
    
    phi = update_V(Vxy_half, phi0)
    phi = update_T(Vkk_full, phi)

    Number_of_iteration = int(t_m/dt)
    widgets = ['time evolution: ', Percentage(), ' ', Bar(marker=RotatingMarker()), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=Number_of_iteration).start()
    for i in range(Number_of_iteration):
        phi = update_V(Vxy_full, phi)
        phi = update_T(Vkk_full, phi)
        pbar.update(i+1)
        if i%50==0:
            plt.clf()
            #ax = plt.gca(projection='3d')
            #ax.plot_surface(X, Y, np.abs(phi), rstride=1, cstride=1, cmap=cm.YlGnBu_r, linewidth = 0)
            plt.contour(X, Y, np.abs(phi),50)
            plt.plot([-2,-2,2,2],[2,-2,2,-2],'ro')
            plt.axis([Lx,Hx,Ly,Hy])
            plt.savefig("./data/%d.jpeg"%i)
            #plt.pause(0.01)
    phi = update_V(Vxy_half, phi)
    pbar.finish()

    return phi

#____main_______
N = 800
L = -500
H = 500
t = 350.0
x = np.linspace(L,H,N)
y = np.linspace(L,H,N)
x0 = -15
sigma = 20
k = 1
phi0 = np.zeros([N,N], dtype = "complex")
for i in range(N):
    for j in range(N):
        phi0[i][j] = np.exp(-0.5*((x[i]-x0)/sigma)**2-0.5*(y[j]/sigma)**2+1j*k*x[i])
plt.figure(figsize = (7, 7))
phi = solver(phi0, L, H, L, H, N, t)
plt.show()

