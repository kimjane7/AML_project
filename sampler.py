from wavefunction import FeedForwardNeuralNetwork
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc


class BruteForce:

    def __init__(self, hamiltonian, maxstep):
        """constructor"""
        self.wavefunction = hamiltonian.wavefunction
        self.hamiltonian = hamiltonian
        self.maxstep = maxstep
    
    
    def sample(self, use_exact=False):
        """return True and change current position if sample is accepted"""
        self.get_trial_sample()
        accepted = False
        if(np.random.sample() < self.calc_acceptance_ratio(use_exact)):
            accepted = True
            self.wavefunction.x = self.trial_x.copy()
        return accepted
        
        
    def get_trial_sample(self):
        """kick one random particle to get new positions"""
        self.trial_x = self.wavefunction.x.copy()
        rand_p = np.random.randint(self.wavefunction.N)
        self.trial_x[rand_p] += self.maxstep*np.random.uniform(-1.0,1.0)


    def calc_acceptance_ratio(self, use_exact):
        """acceptance ratio for Metropolis-Hastings algorithm"""
        
        if use_exact:
            psi = self.hamiltonian.exact_gs_wavefunction(self.wavefunction.x)
            trial_psi = self.hamiltonian.exact_gs_wavefunction(self.trial_x)
        
        else:
            psi = self.wavefunction.calc_psi(self.wavefunction.x)
            trial_psi = self.wavefunction.calc_psi(self.trial_x)
            
        return (trial_psi/psi)**2
        
        
    def plot_samples(self, num_samples, plotfile, use_exact=False):
        """makes density plot for distribution of samples"""
        
        # let sampler to reach equilibrium
        fraction_skip = 0.1
        num_skip_samples = int(fraction_skip*num_samples)
        num_effective_samples = num_samples-num_skip_samples
        for sample in range(num_skip_samples):
            accepted = self.sample(use_exact)
        
        # collect all sampled positions
        x = np.empty(self.wavefunction.N*num_effective_samples)
        for sample in range(num_effective_samples):
            accepted = self.sample(use_exact)
            x[self.wavefunction.N*sample:self.wavefunction.N*(sample+1)] = self.wavefunction.x
        
        plt.figure(figsize=(8,6))
        plt.rc('font', family='serif')
        plt.rc('text', usetex=True)
        plt.xlabel('Position $x$', fontsize=12)
        plt.ylabel(r'Probability distribution $|\Psi|^2$', fontsize=12)
        N = str(self.wavefunction.N)
        nu = str(self.hamiltonian.nu)
        plt.title(r'Distribution of particles in the Calogero model ($N=$ '+N+r', $\nu=$ '+nu+')', fontsize=16)
        sns.set_style("whitegrid")
        sns.kdeplot(x, shade=True)
        plt.savefig(plotfile, format="pdf")
            


"""
class ImportanceSampling:

    def __init__(self, wavefunction):
        
    
    def get_trial_sample():
    
    def calc_acceptance_ratio():
    

"""
    
