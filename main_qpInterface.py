# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 11:27:27 2019
@author: DG70VC
"""

import numpy as np
import qpInterface
import scipy.sparse as sparse

#optimizer = 'sqpmethod'
#optimizer = 'ipopt'
#optimizer = 'qpoases'
optimizer = 'osqp'

optimizerSet = ['qpoases', 'sqpmethod', 'ipopt', 'osqp', 'ecos']

P = np.matrix([[4, 1], [1, 2]])
q = np.array([1, 1])
A = np.matrix([[1, 1], [1, 0], [0, 1]]) # feasible solution
#A = np.matrix([[1, 1], [1, 1], [1, 1]])  # unfeasible solution
l = np.array([1, 0, 0])
u = np.array([1, 0.7, 0.7])

for optimizer in optimizerSet :
    (x, Success) = qpInterface.solveQP(P, q, A, l, u, optimizer)
    print("-----")
    print("Optimizer ="          , optimizer)
    print("> primal solution ="  , x)       # > [0.3, 0.7] 
    print("> Feasible solution =", Success) # > True
