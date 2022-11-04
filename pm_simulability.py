from picos.modeling.problem import Problem, SolutionFailure
import numpy as np
from picos.expressions.variables import HermitianVariable, RealVariable
from picos.expressions.algebra import trace
import qutip as qt

def isPMSimulable(userPOVM):
    problem = Problem()
    
    nEffects=len(userPOVM)
    
    probs=[[RealVariable('c'+str(i)+str(j),lower=0) for i in range(nEffects)] 
           for j in range(nEffects)]
    
    'SUM_{i<j}p_{i,j}=1'
    problem.add_constraint(sum([probs[i][j] for j in range(nEffects) 
                                for i in range(0,j)])==1)
    
    N=[[[HermitianVariable('N'+str(i)+str(j)+str(sign),shape=(2,2)) for sign in [0,1]] 
               for j in range(nEffects)]
               for i in range(nEffects)]
    
    
    problem.add_list_of_constraints([N[i][j][sign]>>0 for j in range(nEffects) 
                                     for i in range(0,j) for sign in [0,1]])
    
    problem.add_constraint(userPOVM[0]==N[0][1][0]+N[0][2][0]+N[0][3][0])
    problem.add_constraint(userPOVM[1]==N[0][1][1]+N[1][2][0]+N[1][3][0])
    problem.add_constraint(userPOVM[2]==N[0][2][1]+N[1][2][1]+N[2][3][0])
    problem.add_constraint(userPOVM[3]==N[0][3][1]+N[1][3][1]+N[2][3][1])
    
    'N_{i,j}^{+} + N_{i,j}^{-}==p_{i,j}I for i<j'
    problem.add_list_of_constraints([N[i][j][0]+N[i][j][1]==probs[i][j]*np.eye(2) 
                                     for j in range(nEffects) for i in range(0,j)])
    
    problem.set_objective("max",0*trace(N[0][1][0]))
    
    try:
        problem.solve(solver="mosek",verbosity=0)
        return True
    except SolutionFailure as s:
        return not(s.code==3)
    
def buildEntangledBasis(theta):
    eta=[1/np.sqrt(3),-1/np.sqrt(3),-1/np.sqrt(3),1/np.sqrt(3)]
    phi=[np.pi/4,7*np.pi/4,3*np.pi/4,5*np.pi/4]
    mPos=[np.sqrt((1+eta[b])/2)*np.exp(-1j*phi[b]/2)*qt.basis(2,0)+np.sqrt((1-eta[b])/2)*np.exp(1j*phi[b]/2)*qt.basis(2,1) 
          for b in range(4)]
    mNeg=[np.sqrt((1-eta[b])/2)*np.exp(-1j*phi[b]/2)*qt.basis(2,0)-np.sqrt((1+eta[b])/2)*np.exp(1j*phi[b]/2)*qt.basis(2,1) 
          for b in range(4)]
    return [(np.sqrt(3)+np.exp(1j*theta))/(2*np.sqrt(2))*qt.tensor(mPos[b],mNeg[b])+
               (np.sqrt(3)-np.exp(1j*theta))/(2*np.sqrt(2))*qt.tensor(mNeg[b],mPos[b])
               for b in range(4)]
        

def testPMSimulabilityForValuesOfTheta(thetas):
    isPMsimulableInThetaRange = False
    
    for theta in thetas:
        basis=buildEntangledBasis(theta)
        
        rhoSA=sum([1/4*qt.ket2dm(basis[b]) for b in range(4)])
    
        rhoA=rhoSA.ptrace([1])
        PiSA=[qt.ket2dm(element) for element in basis]
        
        userPOVM=np.array([(PiSA[x]*qt.tensor(qt.qeye(2),rhoA)).ptrace([0]).full() for x in range(4)])
        
        isPMsimulableInThetaRange=isPMsimulableInThetaRange or isPMSimulable(userPOVM)

    return isPMsimulableInThetaRange

if __name__ == '__main__':
    
    thetas = [i/1000*np.pi/10 for i in range(1000)]
    
    isPMsimulable = testPMSimulabilityForValuesOfTheta(thetas)
    
    if not(isPMsimulable):
        print("The user POVM is not a convex combination of projective measurments for the specified values of theta")
    else:
        print("There exist a theta in the specified set for which the user POVM is PM simulable")
        
