from wavefunction import FeedForwardNeuralNetwork
import numpy as np


class CalogeroSutherland:

    def __init__(self, wavefunction, nu, gamma):
        """constructor"""
        self.wavefunction = wavefunction        # trial wave function
        self.nu = nu                            # interaction parameter
        self.gamma = gamma                      # interaction ramp-up speed hyperparameter
    
    
    def calc_local_energy(self, x, iteration):
        """local energy with gradually introduced interaction potential"""
        EL = self.wavefunction.calc_local_kinetic_energy(x)
        
        sum = 0.0
        for p in range(self.wavefunction.N):
            sum += x[p]**2
        EL += 0.5*sum

        sum = 0.0
        for p in range(self.wavefunction.N-1):
            for q in range(p+1, self.wavefunction.N):
                sum += 1.0/(x[p]-x[q])**2
        
        # ramp up nu at each iteration until desired value of nu is reached
        if self.gamma*iteration > 1.0:
            nu = self.nu
        else:
            nu = self.gamma*iteration*self.nu
        
        EL += nu*(nu-1.0)*sum
        
        return EL
    

    def exact_gs_energy(self):
        """exact ground state energy"""
        return 0.5*self.wavefunction.N \
               +0.5*self.nu*self.wavefunction.N*(self.wavefunction.N-1)
    
    
    def exact_gs_wavefunction(self, x):
        """exact ground state wave function"""
        psi = self.nonint_gs_wavefunction(x);
        for p in range(self.wavefunction.N-1):
            for q in range(p+1, self.wavefunction.N):
                psi *= (abs(x[p]-x[q]))**self.nu
        return psi
    
    def exact_qforce(self, x, p):
        """quantum force on pth particle for exact interacting case"""
        print("need to finish exact qforce calculations")
        return 0
    
    
    def nonint_gs_wavefunction(self, x):
        """ground state wave function for 1D harmonic oscillator"""
        return np.exp(-0.5*np.dot(x,x))
    
    def nonint_qforce(self, x, p):
        """quantum force on pth particle for non-interacting case"""
        return -2.0*x[p]

