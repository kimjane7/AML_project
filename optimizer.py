import numpy as np


class StochasticGradientDescent:

    def __init__(self, wavefunction):
        """constructor"""
    
        self.wavefunction = wavefunction
        self.num_params = self.wavefunction.M*self.wavefunction.N+2*self.wavefunction.M
        self.t = 1
        self.name = 'Stochastic Gradient Descent'
        
    def reset(self):
        """reinitialize optimizer for different task on same wave function"""
    
        self.t = 1
        
        
    def update_params(self, gradient):
        """shift wavefunction parameters"""

        eta = 1.0/self.t
        self.wavefunction.alpha -= eta*gradient
        self.wavefunction.separate(self.wavefunction.alpha)
        self.t += 1





class Adam:

    def __init__(self, wavefunction, learning_rate, beta1, beta2):
        """constructor"""

        self.wavefunction = wavefunction
        self.num_params = self.wavefunction.M*self.wavefunction.N+2*self.wavefunction.M
        self.name = 'Adam'
        
        self.eta = learning_rate
        self.beta1 = beta1                      # exponential decay rate in [0,1)
        self.beta2 = beta2                      # exponential decay rate in [0,1)
        self.m = 0                              # initial first moment vector
        self.v = 0                              # initial second moment vector
        self.t = 1                              # initial time
        self.epsilon = 1e-8                     # prevent division by zero
        
    
    def reset(self):
        """reinitialize optimizer for different task on same wave function"""

        self.m = 0
        self.v = 0
        self.t = 1

        
    def update_params(self, gradient):
        """shift wavefunction parameters"""
        
        # biased moment estimates
        self.m = self.beta1*self.m + (1-self.beta1)*gradient
        self.v = self.beta2*self.v + (1-self.beta2)*gradient**2
        
        # bias-corrected moment estimates
        mhat = self.m/(1-self.beta1**self.t)
        vhat = self.v/(1-self.beta2**self.t)
        
        # update parameters
        self.wavefunction.alpha -= self.eta*mhat/(np.sqrt(vhat)+self.epsilon)
        self.wavefunction.separate(self.wavefunction.alpha)
        self.t += 1
