import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc

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
