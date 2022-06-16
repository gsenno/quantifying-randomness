'''
Created on 14 jun. 2022

@author: gsenno
'''
import unittest
import numpy as np
from pguess import pguess


class Test(unittest.TestCase):

    def testMeasureSigmaZToPlusStateGives1bitOfRandomness(self):
        plusState = 1/np.sqrt(2)*(np.array([1,0])+np.array([0,1]))
        rho = np.outer(plusState,plusState)
        sigmaZ = [[[1,0],[0,0]],[[0,0],[0,1]]]
        cvxDecomp, value = pguess(rho,sigmaZ)
        self.assertTrue(np.isclose(value,0.5))
        
    def testMeasureSigmaXToPlusStateGivesNoRandomness(self):
        plusState = 1/np.sqrt(2)*(np.array([1,0])+np.array([0,1]))
        rho = np.outer(plusState,plusState)
        sigmaX = [rho,np.eye(2)-rho]
        cvxDecomp, value = pguess(rho,sigmaX)
        self.assertTrue(np.isclose(value,1))

    def testMeasureSigmaZToTheMaximallyMixedStateGivesNoRandomness(self):
        rho = 1/2*np.array([[1,0],[0,1]])
        sigmaZ = [[[1,0],[0,0]],[[0,0],[0,1]]]
        cvxDecomp, value = pguess(rho,sigmaZ)
        self.assertTrue(np.isclose(value,1))
        
    def testMeasureUniformlyRandomPOVMToPlusStateGivesNoRandomness(self):
        plusState = 1/np.sqrt(2)*(np.array([1,0])+np.array([0,1]))
        rho = np.outer(plusState,plusState)
        uniformlyRandomPOVM = [1/2*np.eye(2),1/2*np.eye(2)]
        cvxDecomp, value = pguess(rho,uniformlyRandomPOVM)
        self.assertTrue(np.isclose(value,1))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()