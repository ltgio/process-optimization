# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 16:14:03 2019

@author: DG70VC
"""

# import library
import casadi as ca
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

#%% ===========================================================================
# set solver
optimizer = 'ipopt'
#optimizer = 'sqpmethod'
#optimizer = 'qpoases'

# set solver option: 
# detials in : https://www.coin-or.org/Ipopt/documentation/node40.html

# Solver options
opts = {}
if optimizer == 'ipopt' :
    opts["ipopt.tol"] = 1e-8
    opts['ipopt.linear_solver']         = 'mumps'
    opts['ipopt.max_iter']              = 50
    opts['ipopt.print_frequency_iter']  = 10
    opts['ipopt.hessian_approximation'] = 'exact'
    #opts['ipopt.hessian_approximation'] = 'limited-memory'

if optimizer == "sqpmethod":
    opts["qpsol"] = "qpoases"
    opts["qpsol_options"] = {"printLevel":"none"}

#%% ===========================================================================
# Solve quadratic problem QP
# =============================================================================
# Define problem data
P = np.matrix([[4, 1], [1, 2]])
q = np.array([1, 1])
A = np.matrix([[1, 1], [1, 0], [0, 1]]) # feasible solution
#A = np.matrix([[1, 1], [1, 1], [1, 1]])  # unfeasible solution
l = np.array([1, 0, 0])
u = np.array([1, 0.7, 0.7])

# formulate opt
x      = ca.SX.sym('x', P.shape[1])    
f      = x.T @ (0.5*P) @ x + ca.dot(q.T, x) 
g      = A @ x    

# construct and solve nlp
nlp    = {'x': x, 'f': f, 'g': g}
solver = ca.nlpsol('solver', optimizer, nlp, opts)
sol    = solver(lbg = l, ubg = u)
x      = sol['x'].full().flatten()

# print solutions
print("----- QP problem")
print("> success ="        ,  solver.stats()['success'])
print("> primal solution ="      , sol['x'])       
print("> objective at solution =", sol['f'])  
print("> dual solution (x) ="    , sol['lam_x']) 
print("> dual solution (g) ="    , sol['lam_g']) 

#%% ===========================================================================
# Solve Second Order Quadratic Cone Programming (SOCP)
#     maximize    x
#     subject to  x + y + z = 1
#                 x^2 + y^2 <= z^2 
#                 x^2 <= yz   
# =============================================================================
# define problem
x   = ca.SX.sym('x', 3)
f   = -x[0]
g   = []
g.append(x[0] + x[1] + x[2] - 1)      # linear constrain
g.append(x[0]**2 + x[1]**2 - x[2]**2) # second-order cone
g.append(x[0]**2 - x[1]*x[2])         # rotated second-order cone

x0   = np.array([0, 0, 0])
lbg  = np.array([0, -np.inf, -np.inf])
ubg  = np.array([0, 0, 0])
 
# construct nlp     
nlp    = {'x': x, 'f': f, 'g': ca.vertcat(*g)}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
sol    = solver(x0 = x0, lbg = lbg, ubg = ubg)

# print solutions
print("\n ----- SOCP solution")
print("> success ="              , solver.stats()['success'])
print("> primal solution ="      , sol['x'])       
print("> objective at solution =", sol['f'])  
print("> dual solution (x) ="    , sol['lam_x']) 
print("> dual solution (g) ="    , sol['lam_g']) 

#%% ===========================================================================
# Solve Mixed Integer Problem
#    maximize x + 10*y
#    subject to
#             x + 7 y <= 17.5
#             x       <= 3.5
#             x       >= 0
#             y       >= 0
#             x, y integers
# =============================================================================
w  = ca.SX.sym('x', 2)
x  = w[0]
y  = w[1]

f  = -(x + 10*y)
g  = []
g.append( x + 7*y)        
g.append( x)             
g.append(-x)      
g.append(-y)
    
lbg = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
ubg = np.array([ 17.5  ,   3.5  ,  0     ,   0    ])

nlp    = {'x': w, 'f': f, 'g': ca.vertcat(*g)}
#solver = ca.nlpsol('solver', 'ipopt', nlp) # Solve relaxed problem
solver = ca.nlpsol('nlp_solver', 'bonmin', nlp, {"discrete": [True, True]});
sol    = solver(lbg = lbg, ubg = ubg)

print("\n ----- MIP solution")
print("> primal solution ="      , sol['x'])       
print("> objective at solution =", sol['f'])  
print("> dual solution (x) ="    , sol['lam_x']) 
print("> dual solution (g) ="    , sol['lam_g']) 

#%% ===========================================================================
# Solve Non linear Mixer Integer Problem (NMIP)
#      max 0.5*x + y
#      subject to
#          (x - 1)^2 + y^2 <= 3
#          x = [0, 1 ,2, 3] integer
#          y = {0, 2}       real
# =============================================================================

w   = ca.SX.sym('w', 2)
x   = w[0]
y   = w[1]

f   = -(0.5*x + y)
g   = []
g.append((x + 1)**2 + y**2)        
g.append(x)
g.append(y)

w0  = np.array([  0    , 0])
lbg = np.array([-np.inf, 0, 0])
ubg = np.array([  3    , 3, 2])

nlp    = {'x': w, 'f': f, 'g': ca.vertcat(*g)}
#solver = ca.nlpsol('solver', 'ipopt', nlp) # Solve relaxed problem
solver = ca.nlpsol('nlp_solver', 'bonmin', nlp, {"discrete": [False, True]});
sol    = solver(x0 = w0, lbg = lbg, ubg = ubg)

print("-----")
print("> primal solution ="      , sol['x'])       
print("> objective at solution =", sol['f'])  
print("> dual solution (x) ="    , sol['lam_x']) 
print("> dual solution (g) ="    , sol['lam_g']) 

#%% ===========================================================================
# title:     solving Knapsack problom
# benchmark: http://www.mafy.lut.fi/study/DiscreteOpt/DYNKNAP.pdf
# maximize sum_{i}^{n} v_{i} * x_{i}
# subject to
#          sum_{i}^{n} w_{i} * x_{i} <= W
#          x_{i} \in {0, 1}
# where w_{i} is the weight and v_{i} is the value of the item x_{i}
# =============================================================================
df = pd.DataFrame(data = {'Value' : [15,10,9,5], 
                          'Weight': [1, 5, 3,4]})
W  = 8 # max weight

x  =  ca.SX.sym('x', df.shape[0])
f  = -ca.dot(df['Value'].values , x)
g  = []
g.append(ca.dot(df['Weight'].values , x))
g.append(x)    

lbg = np.hstack((np.array([0]) , np.zeros(df.shape[0])))  
ubg = np.hstack((np.array([W]) ,  np.ones(df.shape[0])))       

nlp    = {'x': x, 'f': f, 'g': ca.vertcat(*g)}
#solver = ca.nlpsol('solver', 'ipopt', nlp) # Solve relaxed problem
solver = ca.nlpsol('nlp_solver', 'bonmin', nlp, {"discrete": [True]*df.shape[0]});
sol    = solver(lbg = lbg, ubg = ubg)

print("-----")
print("> primal solution ="      , sol['x'])       
print("> objective at solution =", sol['f'])  
print("> dual solution (x) ="    , sol['lam_x']) 
print("> dual solution (g) ="    , sol['lam_g']) 
print(solver.stats())

#%% ===========================================================================
# Solve Non Linear Program (NLP)
# =============================================================================
# Declare variables
x = ca.MX.sym('x',3,1);      # opt variable
f = x[0]**2 + 2*x[1] - x[2]; # objective
g = x[0] + x[1] + x[2];      # constrains

# costruct nlp
nlp = {'x':x, 'f':f, 'g':g}

# Allocate a solver
solver = ca.nlpsol("solver", optimizer, nlp, opts)

# Solve the NLP
#sol = solver(lbg=0)
x0  = np.array([0,0,0])
lbx = - 1 * np.array([1,1,1])
ubx =   1 * np.array([1,1,1])
lbg = 0
ubg = 0
sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
# Print solution
print("-----")
print("> objective at solution = ", sol["f"])
print("> primal solution = ", sol["x"])
print("> dual solution (x) = ", sol["lam_x"])
print("> dual solution (g) = ", sol["lam_g"])

#%% ===========================================================================
# Solve Parametric Optimization
# =============================================================================

N = 50
rs = np.linspace(1,3,N)

x = ca.SX.sym('x')
y = ca.SX.sym('y')
p = ca.SX.sym('p')

v = ca.vertcat(x,y)

f = (1 - x)**2 + (y - x**2)**2
g = x**2+y**2 - p
nlp = {'x': v, 'p':p ,'f': f, 'g': g}

# Create IPOPT solver object
solver = ca.nlpsol("solver", optimizer, nlp, opts)

# perform pareto optimization
f_pareto = []
x_pareto = []
y_pareto = []

def optimize(rs_idx) :
    sol = solver(x0  = [2.5,3.0], # solution guess
                 lbx = -np.inf,   # lower bound on x
                 ubx =  np.inf,   # upper bound on x
                 lbg = -np.inf,   # lower bound on g
                 ubg =  0,        # upper bound on g
                  p  =  rs_idx)  # parameter
    return (sol["f"], sol["x"])

for rs_idx in rs :
    (fsol, wsol) = optimize(rs_idx)
    f_pareto.append(fsol)
    x_pareto.append(wsol[0])
    y_pareto.append(wsol[1])

# plot   
plt.figure(1)
plt.subplot(211)
plt.title('Pareto Optimization')
plt.plot(rs, f_pareto, '-o')
plt.ylabel('$f_{opt}$')
plt.grid()
plt.subplot(212)
plt.plot(x_pareto, y_pareto, 'r-')
plt.xlabel('$x_{opt}$')
plt.ylabel('$y_{opt}$')
plt.legend(['x','y'])
plt.grid()
plt.show()

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
solver = ca.nlpsol("solver", optimizer, nlp, opts)

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
nlp    = {'x':x, 'f':f, 'g':g}
solver = ca.nlpsol('solver', optimizer, nlp, opts);
sol    = solver(x0 = w0, lbx = lbw, ubx = ubw,lbg = lbg, ubg = ubg)

# Solve the NLP and print solution
print("-----")
print("> objective at solution = ", sol["f"]) # > 0 
print("> primal solution = ", sol["x"])       # > [0, 1, 0]
print("> dual solution (x) = ", sol["lam_x"]) # > [0, 0, 0]
print("> dual solution (g) = ", sol["lam_g"]) # > 0