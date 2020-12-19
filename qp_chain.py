# -*- coding: utf-8 -*-
"""
Model a chain attached to two supports and hanging in between. 
Let us discretise it with N mass points connected by N-1 springs. 
Each mass i has position (yi,zi), i=1,...,N.
The equilibrium point of the system minimises the potential energy.

The potential energy of each spring is
Vi = D_i/2 * ((y_i-y_{i+1})^2 + (z_i-z_{i+1})^2)

The gravitational potential energy of each mass is
Vg_i = m_i*g0*z_i

The total potential energy is thus given by:
Vchain(y,z) = 1/2*sum{i=1,...,N-1} D_i ((y_i-y_{i+1})^2+(z_i-z_{i+1})^2) + g0 * sum{i=1,...,N} m_i * z_i

where y=[y_1,...,y_N] and z=[z_1,...,z_N]

We wish to solve minimize{y,z} Vchain(y, z)

Subject to the piecewise linear ground constraints:
z_i >= zin
z_i - 0.1*y_i >= 0.5
"""

import numpy as np
from casadi import *

# Constants
N   = 40
m_i = 40.0/N
D_i = 70.0*N
g0  = 9.81

zmin = 0.5 # ground
beta = -0.2 # 0.1 # slope constrains
#zmin = -np.inf # unbounded


# Objective function
Vchain = 0

# Variables
x = []

# Variable bounds
lbx = []
ubx = []

# Constraints
g = []

# Constraint bounds
lbg = []
ubg = []

# Loop over all chain elements
for i in range(1, N+1):
   # Previous point
   if i>1:
      y_prev = y_i
      z_prev = z_i

   # Create variables for the (y_i, z_i) coordinates
   y_i = SX.sym('y_' + str(i))
   z_i = SX.sym('z_' + str(i))

   # Add to the list of variables
   x  += [y_i, z_i]
   if i==1: # constraint 1st point
    lbx += [-2., 1.]
    ubx += [-2., 1.]
   elif i==N: # constraint 1st point
    lbx += [ 2., 1.]
    ubx += [ 2., 1.]
   else:
    lbx += [-np.inf, zmin]   # [lby, lbz]
    ubx += [ np.inf, np.inf] # [uby, ubz]

   # Spring potential
   if i > 1:
      Vchain += D_i/2*((y_prev-y_i)**2 + (z_prev-z_i)**2)

   # Graviational potential
   Vchain += g0 * m_i * z_i

   # Slanted ground constraints
   g.append(z_i - beta*y_i)
   lbg.append( 0.5)
   ubg.append(np.inf)

# Formulate QP
qp = {'x':vertcat(*x), 'f':Vchain, 'g':vertcat(*g)}

# Solve with IPOPT
solver = qpsol('solver', 'qpoases', qp, {'sparse':True})
#solver = nlpsol('solver', 'ipopt', qp)

# Get the optimal solution
sol   = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
x_opt = sol['x']
f_opt = sol['f']
print('f_opt = ', f_opt)

# Retrieve the result
Y0 = x_opt[0::2]
Z0 = x_opt[1::2]

# Plot the result
import matplotlib.pyplot as plt
plt.plot(Y0,Z0,'o-')
ys = np.linspace(-2.,2.,100)
zs = 0.5 + beta*ys
plt.plot(ys,zs,'--')
plt.xlabel('y [m]')
plt.ylabel('z [m]')
plt.title('hanging chain QP')
plt.grid(True)
plt.legend(['Chain','z - 0.1y >= 0.5'],loc=9)
plt.show()
