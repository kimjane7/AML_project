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



'''
# for plotting snapshots of trial wave function
iter = 0
snapshots = [0, 10, 100, 200, 1000, 10000]
colors = ['red', 'orange', 'yellowgreen', 'green', 'blue', 'purple']
plt.figure(figsize=(10,8))
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.xlabel(r'Position $x$ of one particle', fontsize=12)
plt.ylabel(r'Positive-definite wave function $\Psi$', fontsize=12)
plt.title('Supervised training for the non-interacting case', fontsize=14)
plt.xlim(-10,10)
plt.ylim(-0.2,1.5)


# plot non-interacting wave function
num_points = 200
x = np.linspace(-10.0,10.0,num_points)
psi_nonint = np.zeros(num_points)
for i in range(num_points):
    psi_nonint[i] = self.hamiltonian.nonint_gs_wavefunction([x[i]])
plt.plot(x, psi_nonint, color='k', label='exact')
'''
'''
# plot snapshots of trial wave function
if cycles == snapshots[iter]:

    psi = np.zeros(num_points)
    for i in range(num_points):
        psi[i] = self.wavefunction.calc_psi([x[i]])
    plt.plot(x, psi, color=colors[iter], label=str(cycles)+' updates')
    iter += 1
    
if iter == len(snapshots):
    
    plt.legend(loc='upper left')
    plt.savefig('supervised_snapshots_20000samples.pdf', format='pdf')
    optimize = False
'''
