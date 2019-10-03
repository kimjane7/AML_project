import numpy as np

class StochasticGradientDescent:

    def __init__(self, wavefunction, learning_rate):
        """constructor"""
    
        self.wavefunction = wavefunction
        self.num_params = self.wavefunction.M*self.wavefunction.N+2*self.wavefunction.M
        self.eta = learning_rate
        
        
    def update_params(self, gradient):
        """shift wavefunction parameters"""
        self.wavefunction.alpha -= self.eta*gradient
        self.wavefunction.W, self.wavefunction.w, self.wavefunction.b \
                                 = self.wavefunction.separate(self.wavefunction.alpha)
