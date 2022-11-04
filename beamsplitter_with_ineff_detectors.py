'''
Created on 4 May 2021

@author: gsenno
'''
from picos.modeling.problem import Problem
import numpy as np
from picos.expressions.variables import HermitianVariable, RealVariable
from itertools import product
from picos.expressions.algebra import trace
import qutip as qt
import matplotlib.pyplot as plt

def pguessExtremals(mu):
    nEffects=4
    rho=1/2*np.outer([0,1,1,0],[0,1,1,0])
    H1=mu*np.asmatrix([[0,0],[0,1]])
    V1=mu*np.asmatrix([[0,0],[0,1]])
    userPovm=[np.kron(np.eye(2)-H1,np.eye(2)-V1),
              np.kron(np.eye(2)-H1,V1),
              np.kron(H1,np.eye(2)-V1),
              np.kron(H1,V1)]
    problem = Problem()
    
    coefs=[RealVariable('c'+str(i),lower=0) for i in range(nEffects)]
    problem.add_constraint(sum(coefs)==1)
    
    variables=[[HermitianVariable('h'+str(a)+str(b)+str(k),shape=(4,4)) for a,b in product([0,1],repeat=2)] for k in range(nEffects)]
    objective=0
    for i in range(nEffects):
        problem.add_list_of_constraints([X>>0 for X in variables[i]])
        problem.add_constraint(sum([variables[j][i] for j in range(nEffects)])==userPovm[i])
        problem.add_constraint(sum(variables[i])==coefs[i]*np.eye(4))
        objective+=trace(variables[i][i]*rho)
    
    problem.set_objective("max",objective)
    problem.solve(solver="mosek",verbosity=0)
    return problem.value

def pguessProjective(mu):
    dims=[2,2,2,2,4]
    psi=1/np.sqrt(2)*((1-mu)*(qt.basis(dims,[0,0,1,0,0])+qt.basis(dims,[1,0,0,0,0])) + \
                      np.sqrt(mu*(1-mu))*(qt.basis(dims,[0,0,1,1,1])+qt.basis(dims,[1,0,0,1,1])+qt.basis(dims,[0,1,1,0,2])+qt.basis(dims,[1,1,0,0,2])) + \
                      mu*(qt.basis(dims,[0,1,1,1,3])+qt.basis(dims,[1,1,0,1,3]))
        )
    rho=qt.ket2dm(psi)
    rho=rho.full()
    
    M=np.kron(np.outer([0,1],[0,1]),np.outer([0,1],[0,1]))
    userPovm=[np.kron(np.eye(4)-M,np.eye(4)-M),
              np.kron(np.eye(4)-M,M),
              np.kron(M,np.eye(4)-M),
              np.kron(M,M)]
    return _pguessProjective(mu,rho,userPovm)


def _pguessProjective(mu,rho,userPovm):
    nEffects=len(userPovm)

    problem = Problem()

    eveMsrmt=[HermitianVariable('E'+str(i),shape=(4,4)) for i in range(nEffects)]
    
    problem.add_list_of_constraints([E>>0 for E in eveMsrmt])
    problem.add_constraint(sum(eveMsrmt)==np.eye(4))
    objective=sum([trace(userPovm[i]@eveMsrmt[i]*rho) for i in range(nEffects)])
    problem.set_objective('Max',objective)
    problem.solve(solver="mosek",verbosity=0)
    return problem.value


if __name__ == '__main__':
    fileName='data.txt'
    mu=0.1

    with open(fileName, 'w') as the_file:
        for sl in range(0,101,1):
            mu=sl/100
            print((mu,pguessExtremals(mu),pguessProjective(mu)),file=the_file)
  
          
    with open(fileName, 'r') as the_file:
        res=[]
        for line in the_file:
            item = line.strip('()\n').split(',')
            res.append((float(item[0]),float(item[1]),float(item[2])))
      
    fig, ax = plt.subplots()
    ax.plot([epsilon for (epsilon,_,_) in res], [m1 for (_,m1,_) in res],color='r')
    ax.plot([epsilon for (epsilon,_,_) in res], [m1 for (_,_,m1) in res],color='b')

    ax.set(xlabel=r'$\mu$', ylabel=r'')
    ax.legend( [r'$p_{{\rm guess}}(X|E,|\psi\rangle_{12},M_\mu)$',r'$f(\mu)$ (Eq. (7))'], fontsize = 14, loc = 'upper right')
    ax.grid()
   
    plt.show()

