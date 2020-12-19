# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 11:25:02 2019
@title : QP solver
@author: DG70VC
"""

import numpy  as np
import scipy.sparse as sparse
import os, sys

# Option
#optimizer = 'sqpmethod'
#optimizer = 'ipopt'
#optimizer = 'qpoases'
#optimizer = 'osqp'
#optimizer = 'ecos'

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def checkFeasibility(P, A, l, u) :            
    import casadi as ca
    if isinstance(P, sparse.csc.csc_matrix) :
        P = sparse.csr_matrix.todense(P)
    if isinstance(A, sparse.csc.csc_matrix) :
        A = sparse.csr_matrix.todense(A)
    x      = ca.SX.sym('x', P.shape[1])
    solver = ca.nlpsol('solver', 'ipopt', {'x': x, 'f': 0, 'g': A @ x})
    with HiddenPrints():
        solver(lbg = l, ubg = u)
    # check feasibility
    if solver.stats()['success'] == True :
        Success  = True
    else :
        Success = False
    return Success

def solveQP(P, q, A, l, u, optimizer) :    
    # Common setting
    max_iter   = 500000 
    abstol     = 1e-6 
    reltol     = 1e-6 
    verbose    = False
    warm_start = False
    
    # -------------------------------------------------------------------------
    if (optimizer == 'osqp') :
        import osqp
        # convert P and A to sparse matrix
        if not(isinstance(P, sparse.csc.csc_matrix)) :
            P = sparse.csc_matrix(P)
        if not(isinstance(A, sparse.csc.csc_matrix)) :
            A = sparse.csc_matrix(A)
        solver = osqp.OSQP()
        solver.setup(P, q, A, l, u, polish  = 1, warm_start = warm_start, max_iter = max_iter, verbose = verbose, eps_abs = abstol, eps_rel = reltol)
        sol = solver.solve()
        if sol.info.status == 'solved' :
            Success = True
        else :
            Success = False
        solution = (sol.x, Success)      
        4
    # -------------------------------------------------------------------------
    elif (optimizer == 'ipopt' or optimizer == 'sqpmethod' or optimizer == 'qpoases') : 
        
        if checkFeasibility(P, A, l, u) :
            import casadi as ca
            Success = True
            if isinstance(P, sparse.csc.csc_matrix) :
                P = sparse.csr_matrix.todense(P)
            if isinstance(A, sparse.csc.csc_matrix) :
                A = sparse.csr_matrix.todense(A)
            # -----------------------------------------------------------------    
            if (optimizer == 'ipopt' or optimizer == 'sqpmethod') :
                # set option
                opts = {}
                if optimizer == 'ipopt' :          
                    opts['ipopt.linear_solver']           = 'mumps'     
                    opts['ipopt.hessian_approximation']   = 'exact'    
                    opts['ipopt.dual_inf_tol']            =  1          
                    opts['ipopt.constr_viol_tol']         =  0.000001    
                    opts['ipopt.compl_inf_tol']           =  0.000001    
                    opts['ipopt.nlp_scaling_method']      = 'gradient-based' 
                    opts["ipopt.tol"]                     = abstol      
                    opts['ipopt.max_iter']                = max_iter
                    opts['ipopt.mehrotra_algorithm']      = 'yes'      
                    opts['ipopt.warm_start_init_point']   = 'no'
                    opts['ipopt.inf_pr_output']           = 'original'
                    opts['ipopt.print_timing_statistics'] = 'no'                   

                # problem formulation
                x      = ca.SX.sym('x', P.shape[1])    
                f      = x.T @ (0.5*P) @ x + ca.dot(q.T, x) 
                g      = A @ x    
                nlp    = {'x': x, 'f': f, 'g': g}
                solver = ca.nlpsol('solver', optimizer, nlp, opts)
                with HiddenPrints():
                    sol  = solver(lbg = l, ubg = u)                    
                x        = np.round(sol['x'].full().flatten(),5)
                solution = (x, Success) 

            # -----------------------------------------------------------------
            if optimizer == 'qpoases' :
                opts = {}
                opts['sparse'] = True
                x        = ca.SX.sym('x', P.shape[1])    
                f        = x.T @ (0.5*P) @ x + ca.dot(q.T, x) 
                g        = A @ x    
                nlp      = {'x': x, 'f': f, 'g': g}
                solver   = ca.qpsol('solver', optimizer, nlp, opts)
                with HiddenPrints():
                    sol      = solver(lbg = l, ubg = u)
                x        = sol['x'].full().flatten()
                Success  = True
                solution = (x, Success) 
        else :
            Success = False
            x = np.nan*np.ones(P.shape[1])
            solution = (x, Success) 

    # -------------------------------------------------------------------------   
    elif (optimizer == 'ecos') : 
        import ecos       
        A = np.vstack((A, -A ))
        b = np.hstack((u, -l))       
        n = P.shape[1]  
        # precondition
        scale = max(abs(q))
        if (scale==0) :
            scale = 1
        f     = q/scale
        H     = P/scale               
        W = np.linalg.cholesky(H) # H must be positive definite
        # set up SOCP problem
        c = np.hstack((np.zeros(n), 1)) 
        # create second-order cone constraint for objective function
        fhalf      = f / np.sqrt(2);
        zerocolumn = np.zeros(W.shape[0])
        Gquad      = np.vstack(( np.hstack((fhalf.T , -1/np.sqrt(2))),
                                 np.hstack((   -W   ,  np.array([zerocolumn]).T)),
                                 np.hstack((-fhalf.T,  1/np.sqrt(2)))))
        hquad = np.hstack((1/np.sqrt(2), zerocolumn, 1/np.sqrt(2)))
        dims = {}
        G = np.vstack((np.hstack((A, np.array([np.zeros(A.shape[0])]).T)) , Gquad))  
        h = np.hstack((b, hquad))
        dims['l'] = A.shape[0]
        dims['q'] = [W.shape[0]+2]   
        if not(isinstance(G, sparse.csc.csc_matrix)) :
            G = sparse.csc_matrix(G)
        solution = ecos.solve(c,G,h,dims, abstol = abstol, reltol = reltol, verbose = verbose, max_iters = max_iter)
                
        if solution['info']['infostring'] == 'Optimal solution found' :
            Success = True
            x       = np.round(solution['x'][:-1],5)
        else :
            Success = False
            x = np.nan*np.ones(P.shape[1])
        solution = (x, Success)
    # -------------------------------------------------------------------------                
    else :
        raise ValueError("""Optimizer not found. Please select among the following choice:
                            -> 'sqpmethod' | 'ipopt' | 'qpoases' | 'osqp' | 'ecos' """)

    return solution