import osqp
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import casadi as ca

# 1st exercise
# ========= OSQP ===========

m = 1

# Define problem data
P = sparse.csc_matrix([[2, 0], [0, 2]])
q = np.array([0, 1])


# Create an OSQP object
prob = osqp.OSQP()
# Setup workspace
prob.setup(P, q)

# Solve problem
res = prob.solve()

print(res.x)


# ========= casadi ===========

import casadi as ca

x = ca.SX.sym('x', 2)
x1 = x[0]
x2 = x[1]
m=1

f = x1**2 + x2**2 + m*x2

# construct nlp
nlp    = {'x': x, 'f': f}
solver = ca.nlpsol('solver', 'ipopt', nlp)
sol    = solver()

# print solutions
print("\n ----- SOCP solution")
print("> success ="              , solver.stats()['success'])
print("> primal solution ="      , sol['x'])
print("> objective at solution =", sol['f'])



# 2nd exercise
# ===============================

import osqp
import numpy as np
from scipy import sparse

# Define problem data
P = sparse.csc_matrix([[8, 0, 0], [0, 4, 0], [0, 0, 3]])
q = np.array([1, 1, 0])
#A = sparse.csc_matrix([[1, 1], [1, 0], [0, 1]])
#l = np.array([1, 0, 0])
#u = np.array([1, 0.7, 0.7])

# Create an OSQP object
prob = osqp.OSQP()

# Setup workspace and change alpha parameter
prob.setup(P, q, alpha=1.0)

# Solve problem
res = prob.solve()

print(res.x)


# ========= casadi ===========

import casadi as ca

x = ca.SX.sym('x', 3)
x1 = x[0]
x2 = x[1]
x3 = x[2]

f = 4*x1**2 + 2*x2**2 + x1 + x2 + x3**2

# construct nlp
nlp    = {'x': x, 'f': f}
solver = ca.nlpsol('solver', 'ipopt', nlp)
sol    = solver()

# print solutions
print("\n ----- SOCP solution")
print("> success ="              , solver.stats()['success'])
print("> primal solution ="      , sol['x'])
print("> objective at solution =", sol['f'])


# ==============================

# 2nd exercise (constrained)
# ===============================

import osqp
import numpy as np
from scipy import sparse

# Define problem data
P = sparse.csc_matrix([[8, 0, 0], [0, 4, 0], [0, 0, 3]])
q = np.array([1, 1, 0])
A = sparse.csc_matrix([[1, 1, 0], [1, 0, 0], [0, 0, 1]])
l = np.array([1, 0, 1])
u = np.array([1, 2, 3])

# Create an OSQP object
prob = osqp.OSQP()

# Setup workspace and change alpha parameter
prob.setup(P, q, A, l, u, alpha=1.0)

# Solve problem
res = prob.solve()

print(res.x)


# ========= casadi ===========

import casadi as ca

x = ca.SX.sym('x', 3)
x1 = x[0]
x2 = x[1]
x3 = x[2]

f = 4*x1**2 + 2*x2**2 + x1 + x2 + x3**2

g = []
g.append(x1 + x2)  # linear constrain
g.append(x1)  # second-order cone
g.append(x3)  # rotated second-order cone

x0 = np.array([0, 0, 0])
lbg = np.array([1, 0, 1])
ubg = np.array([1, 2, 3])

# construct nlp
nlp = {'x': x, 'f': f, 'g': ca.vertcat(*g)}
solver = ca.nlpsol('solver', 'ipopt', nlp, )
sol = solver(x0=x0, lbg=lbg, ubg=ubg)

# print solutions
print("\n ----- SOCP solution")
print("> success ="              , solver.stats()['success'])
print("> primal solution ="      , sol['x'])
print("> objective at solution =", sol['f'])



#%% ===========================================================================
# Solve Rosenbrock (unconstrained)
# =============================================================================
# Plot non linear function ----------------------------------------------------
[X0,X1] = np.meshgrid(np.linspace(-3.,3.,1000), np.linspace(-3.,3.,1000))
F = np.exp(-X0**2 - X1**2) * np.sin(4 * (X0 + X1 + X0*X1**2))

# Plot the function
plt.clf()
plt.contour(X0,X1,F)
plt.colorbar()
plt.jet()
plt.xlabel("x0")
plt.ylabel("x1")
plt.ylim([-3,3])
plt.xlim([-3,3])

# define problem
x = ca.SX.sym('x',2)
f = ca.exp(-x[0]**2 - x[1]**2 ) * ca.sin(4*(x[0] + x[1] + x[0]*x[1]**2 ))

# construct nlp
nlp = {'x': x,'f': f}
solver = ca.nlpsol("solver", 'ipopt', nlp)

# Gradient of f
F_grad = ca.Function('f', [x], [ca.gradient(f, x)], ['x'], ['gradient'])
print(F_grad)

## Solve for three different starting points
summary = [("GUESS", "SOLUTION", "SOLVER STATUS")]
for x_guess in [[0, 0], [0.9, 0.9], [-0.9, -0.9]]:
    # Solve the NLP and get output
    sol   = solver(x0 = x_guess)
    x_opt = sol['x'].full().flatten()
    summary.append((x_guess, x_opt, solver.stats()['return_status']))
    plt.plot([x_guess[0],x_opt[0]], [x_guess[1],x_opt[1]],'ro-')

print('SUMMARY:')
for (x0, x_opt, status) in summary:
    print("%20s , %20s , %20s" % (x0, x_opt, status))
    if not(isinstance(x_opt, str)):
        # Check optimality
        print("  Gradient: %20s" % (F_grad(x_opt)))

# Show plot
plt.axis([-3,3,-3,3])
plt.show()

#%% ===========================================================================
# Solve Rosenbrock (constrained)
# =============================================================================
x   = ca.MX.sym('x',3,1);
f   = x[0]**2 + 100*x[2]**2;
g   = x[2] + (1-x[0])**2 - x[1];

# set i.c. and bounds
w0  = [2.5,3.0,0.75] # initial guess array
lbw = -np.inf        # lower bound solution
ubw =  np.inf        # upper bound solution
lbg = 0;             # lower bound inequality array
ubg = 0;             # upper bound inequality array

# construct nlp
nlp    = {'x':x, 'f':f, 'g':g} # with constrains
nlp    = {'x':x, 'f':f} # without constrains
solver = ca.nlpsol('solver', 'ipopt', nlp);
sol    = solver(x0 = w0, lbx = lbw, ubx = ubw,lbg = lbg, ubg = ubg)

# Solve the NLP and print solution
print("-----")
print("> objective at solution = ", sol["f"]) # > 0
print("> primal solution = ", sol["x"])       # > [0, 1, 0]
print("> dual solution (x) = ", sol["lam_x"]) # > [0, 0, 0]
print("> dual solution (g) = ", sol["lam_g"]) # > 0