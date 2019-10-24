from wavefunction import FeedForwardNeuralNetwork
import numpy as np


class Calogero:

    def __init__(self, wavefunction, nu):
        """constructor"""
        self.wavefunction = wavefunction        # trial wave function
        self.nu = nu                            # interaction parameter

    
    def calc_local_energy(self, x):
        """local energy"""
        EL = self.wavefunction.calc_local_kinetic_energy(x)
        
        sum = 0.0
        for p in range(self.wavefunction.N):
            sum += x[p]**2
        EL += 0.5*sum
        
        sum = 0.0
        for p in range(self.wavefunction.N-1):
            for q in range(p+1, self.wavefunction.N):
                sum += 1.0/(x[p]-x[q])**2
        EL += self.nu*(self.nu-1)*sum
        return EL
    

    def exact_gs_energy():
        """exact ground state energy"""
        return 0.5*self.wavefunction.N \
               +0.5*self.nu*self.wavefunction.N*(self.wavefunction.N-1)
    
    
    def exact_gs_wavefunction(x):
        """exact ground state wave function"""
        sum = 0.0
        for p in range(self.wavefunction.N):
            sum += x[p]**2
        psi = np.exp(-0.5*sum)
        
        prod = 1.0
        for p in range(self.wavefunction.N-1):
            for q in range(p+1, self.wavefunction.N):
                prod *= (abs(x[p]-x[q]))**self.nu
        psi *= prod
        return psi
