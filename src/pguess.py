from picos.modeling.problem import Problem
import numpy as np
from picos.expressions.variables import HermitianVariable, RealVariable
from picos.expressions.algebra import trace, sum
from picos.modeling.solution import SS_INFEASIBLE

def pguess(aState,aPOVM):
    
    if _isExtremalState(aState):
        if _isExtremalPOVM(aPOVM):
            nElementsInCvxDecomp = len(aPOVM)
            vPguess = max([np.trace(anEffect@aState) for anEffect in aPOVM])
            probabilities = np.zeros(nElementsInCvxDecomp)
            probabilities[0]=1
            return [[probabilities,[aPOVM for _ in range(nElementsInCvxDecomp)]],vPguess]
            return max([np.trace(anEffect@aState) for anEffect in aPOVM])
        else:
            return _sdpForMeasurement(aState,aPOVM)
    else:
        if _isExtremalPOVM(aPOVM):
            return _sdpForState(aState, aPOVM)
        else:
            raise Exception('pguess when both the state and the measurement are not extremal is not implemented') 

def _isExtremalState(aState):
    return np.isclose(np.trace(aState@aState),1)

def _sdpForState(aState,anExtremalPOVM):
    dim = len(aState)
    nOutcomes=len(anExtremalPOVM)
    
    'SDP'
    problem = Problem()
    variables=[HermitianVariable('phi_'+str(a),shape=(dim,dim)) for a in range(nOutcomes)]
    
    'Constraints'
    problem.add_list_of_constraints([X>>0 for X in variables])
    problem.add_constraint(sum(variables)==aState)
    objective=sum([trace(anExtremalPOVM[i]*variables[i]) for i in range(dim)])
     
    problem.set_objective("max",objective)
    problem.solve()

    unnormalizedStates = [np.array(v.value_as_matrix) for v in variables]
    return [(
            [np.trace(unnormalizedState) for unnormalizedState in unnormalizedStates],
            [unnormalizedState/np.trace(unnormalizedState) for unnormalizedState in unnormalizedStates]
            ),
            problem.value]



def _isExtremalPOVM(aPOVM):
    dim = len(aPOVM[0])
    nOutcomes = len(aPOVM)

    problem = Problem()
     
    'Variables'
    vLambda=RealVariable('lambda',lower=0)
    onePOVM=[HermitianVariable('fst'+str(a),shape=(dim,dim)) for a in range(nOutcomes)]
    anotherPOVM=[HermitianVariable('snd'+str(a),shape=(dim,dim)) for a in range(nOutcomes)] 
            
    'Constraints'
    problem.add_list_of_constraints([X>>0 for X in onePOVM])
    problem.add_constraint(sum(onePOVM)==vLambda*np.eye(dim))

    problem.add_list_of_constraints([X>>0 for X in anotherPOVM])
    problem.add_constraint(sum(anotherPOVM)==(1-vLambda)*np.eye(dim))
    
    problem.add_list_of_constraints([aPOVM[i]==onePOVM[i]+anotherPOVM[i] for i in range(nOutcomes)])
        
    objective=0*trace(onePOVM[0][0])
    
    try:
        problem.set_objective("max",objective)
        problem.solve()
        onePOVM = _extractPOVMFromListOfOptimizationVariable(onePOVM)
        anotherPOVM = _extractPOVMFromListOfOptimizationVariable(anotherPOVM)
        vLambda=vLambda.value
        return (np.allclose(onePOVM,vLambda*np.array(aPOVM)) 
                or np.allclose(anotherPOVM,(1-vLambda)*np.array(aPOVM)))
    except:
        return problem.status==SS_INFEASIBLE

def _extractPOVMFromListOfOptimizationVariable(aListOfHermitianVariables):
    return [np.array(effect.value_as_matrix) for effect in aListOfHermitianVariables]

def _sdpForMeasurement(aPureState,aPOVM):
    dim = len(aPureState)
    nOutcomes = len(aPOVM)
    
    problem = Problem()
     
    'Variables'
    probs=[RealVariable('c'+str(i),lower=0) for i in range(nOutcomes)]
    variables=[[HermitianVariable('h'+str(a)+str(k),shape=(dim,dim)) for a in range(nOutcomes)] 
               for k in range(nOutcomes)]

    'Constraints'
    problem.add_constraint(sum(probs)==1)
    for i in range(nOutcomes):
        problem.add_list_of_constraints([X>>0 for X in variables[i]])
        problem.add_constraint(sum([variables[j][i] for j in range(nOutcomes)])==aPOVM[i])
        problem.add_constraint(sum(variables[i])==probs[i]*np.eye(dim))
    
    objective=sum([trace(variables[i][i]*aPureState) for i in range(nOutcomes)])
    
    problem.options["rel_ipm_opt_tol"]=10**-11
    problem.options["rel_prim_fsb_tol"]=10**-11
    problem.options["rel_dual_fsb_tol"]=10**-11
    problem.options["max_footprints"]=None

    
    problem.set_objective("max",objective)
    problem.solve()
    return [(
            [p.value for p in probs],
            [_extractPOVMFromListOfOptimizationVariable(variables[i]) for i in range(nOutcomes)]
            ),
            problem.value]
