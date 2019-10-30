import numpy as np

class StochasticGradientDescent:

    def __init__(self, wavefunction):
        """constructor"""
    
        self.wavefunction = wavefunction
        self.num_params = self.wavefunction.M*self.wavefunction.N+2*self.wavefunction.M
        self.iterations = 1
        
        
    def update_params(self, gradient):
        """shift wavefunction parameters"""

        eta = 0.1/self.iterations
        self.wavefunction.alpha -= eta*gradient
        self.wavefunction.W, self.wavefunction.b, self.wavefunction.w \
                                 = self.wavefunction.separate(self.wavefunction.alpha)
        self.iterations += 1
