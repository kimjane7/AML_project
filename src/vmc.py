import numpy as np

class VariationalMonteCarlo:

    def __init__(self, optimizer, sampler, hamiltonian, num_samples, tolerance, filename):
        """constructor"""
        self.optimizer = optimizer
        self.sampler = sampler
        self.hamiltonian = hamiltonian
        self.wavefunction = self.hamiltonian.wavefunction
        self.num_params = self.optimizer.num_params
        self.num_samples = num_samples
        self.tolerance = tolerance
        self.filename = filename
    
    def minimize_energy(self):
        """optimizes of wave function parameters
           and calculates ground state energy"""
        
        optimize = True
        cycles = 0
        
        while optimize:
            
            cycles += 1
            num_accepted = 0
            num_effective_samples = 0
            
            self.EL_mean = 0.0
            self.EL2_mean = 0.0
            self.gradient_logpsi_mean = np.zeros(self.num_params)
            self.EL_gradient_logpsi_mean = np.zeros(self.num_params)
            
            
            
            
            
        
        
    def estimate_gradient(self):
        
        return grad
    
