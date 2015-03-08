import numpy as np
import matplotlib.pyplot as plt

def V(x):
    return -5*np.exp(-0.5*np.abs(x))+1/(np.abs(x)+0.01)

def update_V(dt, phi, x):
    return np.exp(-1j*dt*V(x))*phi

def update_T(dt, phi, x):
    N = len(x)
    k = np.linspace(-20,20,N)
    psi = np.fft.fft(phi)
    for i in range(0, len(psi)):
        psi[i] = np.exp(-1j*dt*(k[(i+int(0.5*N))%N])**2)*psi[i]
    phi = np.fft.ifft(psi)
    return phi


def solver(phi, x, t_m):
    N_x = len(x)
    dx = x[1]-x[0]
    dt = 0.01
    phi = update_V(dt/2., phi, x)
    phi = update_T(dt, phi, x)
    
    N_max = int(t_m/dt)
    for i in range(N_max):
        phi = update_V(dt, phi, x)
        phi = update_T(dt, phi, x)
        if i%5==0:
            plt.clf()
            plt.plot(x, V(x),'r-')
            plt.plot(x, np.abs(phi),'b-')
            plt.axis([-20,20,-2,2])
            plt.pause(0.02)
    phi = update_V(dt/2., phi, x)
    return phi

#____main_______
x = np.linspace(-20,20,2000)
x0 = -10
sigma = 2
phi0 = np.exp(-0.5*((x-x0)/sigma)**2+1j*x*40)
plt.figure(figsize = (15, 5))
phi = solver(phi0,x,15.0)
plt.show()

