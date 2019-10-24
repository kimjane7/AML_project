from wavefunction import FeedForwardNeuralNetwork
import numpy as np


class BruteForce:

    def __init__(self, wavefunction, maxstep):
        """constructor"""
        self.wavefunction = wavefunction
        self.maxstep = maxstep
    
    
    def sample(self):
        """return True and change current position if sample is accepted"""
        self.get_trial_sample()
        accepted = False
        if(np.random.sample < self.calc_acceptance_ratio()):
            accepted = True
            self.wavefunction.x = self.trial_x
        return accepted
        
        
    def get_trial_sample(self):
        """kick one random particle to get new positions"""
        self.trial_x = self.wavefunction.x
        rand_p = np.random.randint(self.wavefunction.N)
        self.trial_x[rand_p] += self.maxstep*np.random.uniform(-1.0,1.0)


    def calc_acceptance_ratio(self):
        """acceptance ratio for Metropolis-Hastings algorithm"""
        psi = self.wavefunction.calc_psi(self.wavefunction.x)
        trial_psi = self.wavefunction.calc_psi(self.trial_x)
        return (trial_psi/psi)**2
        


"""
class ImportanceSampling:

    def __init__(self, wavefunction):
        
    
    def get_trial_sample():
    
    def calc_acceptance_ratio():
    

"""
    
