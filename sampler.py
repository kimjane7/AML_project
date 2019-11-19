from wavefunction import FeedForwardNeuralNetwork
import numpy as np



class BruteForce:

    def __init__(self, hamiltonian, maxstep):
        """constructor"""
        self.wavefunction = hamiltonian.wavefunction
        self.hamiltonian = hamiltonian
        self.maxstep = maxstep
    
    
    def sample(self):
        """sample from non-interacting ground state,
           exact ground state, or trial wave function """
        self.get_trial_sample()
        accepted = False
        if(np.random.sample() < self.calc_acceptance_ratio()):
            accepted = True
            self.wavefunction.x = np.sort(self.trial_x)
        return accepted
        
        
    def get_trial_sample(self):
        """kick one random particle to get new positions"""
        self.trial_x = self.wavefunction.x.copy()
        rand_p = np.random.randint(self.wavefunction.N)
        self.trial_x[rand_p] += self.maxstep*np.random.uniform(-1.0,1.0)


    def calc_acceptance_ratio(self):
        """acceptance ratio for Metropolis-Hastings algorithm"""
        
        psi = self.wavefunction.calc_psi(self.wavefunction.x)
        trial_psi = self.wavefunction.calc_psi(self.trial_x)
            
        return (trial_psi/psi)**2



class ImportanceSampling:

    def __init__(self, hamiltonian, timestep):
    
        self.hamiltonian = hamiltonian
        self.wavefunction = self.hamiltonian.wavefunction
        self.timestep = timestep
        self.diff_const = 0.5

    def sample(self):
        """return True and change current position if sample is accepted"""
        self.get_trial_sample()
        accepted = False
        if(np.random.sample() < self.calc_acceptance_ratio()):
            accepted = True
            self.wavefunction.x = np.sort(self.trial_x)
        return accepted
        

    def get_trial_sample(self, option):
        """kick one random particle using quantum force to get new positions"""
        self.trial_x = self.wavefunction.x.copy()
        self.rand_p = np.random.randint(self.wavefunction.N)
        self.qforce = self.wavefunction.calc_qforce(self.wavefunction.x, self.rand_p)
        self.trial_x[self.rand_p] += self.diff_const*self.timestep*self.qforce \
                                     + np.random.normal()*np.sqrt(self.timestep)
    
        
    
    def calc_acceptance_ratio(self, option):
        """acceptance ratio for importance sampling algorithm"""
        
        psi = self.wavefunction.calc_psi(self.wavefunction.x)
        trial_psi = self.wavefunction.calc_psi(self.trial_x)
        
        self.trial_qforce = self.wavefunction.calc_qforce(self.trial_x, self.rand_p)
        
        greens = 0.5*(self.wavefunction.x[self.rand_p]-self.trial_x[self.rand_p])*(self.trial_qforce+self.qforce) \
                 + 0.25*self.diff_const*self.timestep*(self.qforce**2-self.trial_qforce**2)
            
        return np.exp(greens)*(trial_psi/psi)**2

