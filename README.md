Rep_kwy_01
==========
This is a 1D Quantum scattering solver using third order split operator mehtod,
f(x,t+dt) = exp(-i(T+V)dt)f(x,t)
          = exp(-iV*dt/2) exp(-iT*dt) exp(-iV*dt/2) f(x,t) + O(dt^3)
Then,
f(x,t) = exp(-iV*dt/2) * (exp(-iT*dt) exp(-iV*dt))^N * exp(-iT*dt) exp(-iV*dt/2) f(x,0)

with the operation done in either spatial space or in momentum space by fourier transformation
f1 = exp(-iV*dt) f0
f2 = ifft(exp(-iT*dt)fft(f0))
