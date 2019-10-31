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
           
        # get weights and biases for non-interacting case
        print('Training for the non-interacting case...')
        self.train_nonint_case()
        print('Done.')
        
        # train wave function for interacting case
        print('Training for the interacting case of nu = ', self.hamiltonian.nu, '...')
        self.train_int_case()
        print('Done.')
        
            
    def train_nonint_case(self):
        """supervised training of initial weights and biases of wave function using known
           wave function for the non-interacting case of particles in 1D harmonic oscillator"""
           
        optimize = True
        cycles = 0
        print('{:<10s}{:<20s}{:<20s}{:<20s}{:<20s}'.format('cycles', 'avg EL', 'var EL', '||gradient||', 'ratio accepted samples'))
        
        while optimize:
        
            self.
           
    def train_int_case(self):
        """reinforcement learning of ground state wave function in interacting case"""
        
        
        CALC OVERLAP DURING ITERATIONS - FIDELITY?
        
        optimize = True
        cycles = 0
        print('{:<10s}{:<20s}{:<20s}{:<20s}{:<20s}'.format('cycles', 'avg EL', 'var EL', '||gradient||', 'ratio accepted samples'))
        
        while optimize:
            
            self.estimate_gradient()
            self.optimizer.update_params(self.gradient)
            cycles += 1
            
            print('{:<10d}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.15}'.format(cycles, self.avg_EL, self.var_EL, np.linalg.norm(self.gradient), self.ratio_accepted))
            
            if np.linalg.norm(gradient) < self.tolerance:
                optimize = False
        
        
    def estimate_gradient(self):
        """estimate gradient of average local energy
           with respect to wave function parameters"""

        num_accepted = 0
        self.avg_EL = 0.0
        avg_EL2 = 0.0
        avg_gradient_logpsi = np.zeros(self.num_params)
        avg_EL_gradient_logpsi = np.zeros(self.num_params)
        
        # let sampler to reach equilibrium
        fraction_skip = 0.1
        num_skip_samples = int(fraction_skip*self.num_samples)
        num_effective_samples = self.num_samples-num_skip_samples
        for sample in range(num_skip_samples):
            accepted = self.sampler.sample()
            
        # take samples to estimate gradient
        for sample in range(num_effective_samples):
        
            # count accepted samples
            accepted = self.sampler.sample()
            if accepted:
                num_accepted += 1
            
            # calculate local energy and gradient of wave function
            EL = self.hamiltonian.calc_local_energy(self.wavefunction.x)
            gradient_logpsi = self.wavefunction.calc_gradient_logpsi(self.wavefunction.x)
            
            # add up values for expectation values
            self.avg_EL += EL
            avg_EL2 += EL**2
            avg_gradient_logpsi += gradient_logpsi
            avg_EL_gradient_logpsi += EL*gradient_logpsi
        
        # calculate expectation values
        self.avg_EL /= num_effective_samples
        avg_EL2 /= num_effective_samples
        avg_gradient_logpsi /= num_effective_samples
        avg_EL_gradient_logpsi /= num_effective_samples
        
        # calculate variance and ratio accepted
        self.var_EL = avg_EL2-self.avg_EL**2
        self.ratio_accepted = float(num_accepted)/num_effective_samples
        
        # calculate gradient
        self.gradient = 2.0*(avg_EL_gradient_logpsi-self.avg_EL*avg_gradient_logpsi)
    
