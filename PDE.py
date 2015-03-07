import numpy as np
import matplotlib.pyplot as plt

def S(x):
    return 0.3*np.exp(-(x-5)*(x-5))

#1D diffusion solver
def solver(f, t_max):
    N_max_x = len(f)
    x = np.linspace(0,10,N_max_x)
    dx = 10./N_max_x
    dt = dx*dx/2.
    f_temp = np.copy(f)
    N_max_t = int(t_max/dt)
    
    for n in range(N_max_t):
        for i in range(1, N_max_x-1):
            f_temp[i] = f[i] + (f[i+1]+f[i-1]-2*f[i])/dx/dx*dt + S(x[i])*dt
        f = np.copy(f_temp)
        if n%20==0:
            plt.clf()
            plt.plot(x, f,'b-')
            plt.axis([0,10,0,1])
            plt.pause(0.1)   
    return 0

#___Main_______
x = np.linspace(0,10,200)
sigma = 0.5
x0 = 3
f = np.exp(-(x-x0)*(x-x0)/sigma/sigma)
r = solver(f, 2)
