import numpy as np
from wavefunction import FeedForwardNeuralNetwork
from hamiltonian import CalogeroSutherland
from sampler import ImportanceSampling
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import rc

plt.rc('font', family='serif')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


def plot_supervised_snapshots_N1(M):

    file = 'initialstates/N1_M'+str(M)+'.txt'
    statefile = open(file, 'r')
    snapshots = statefile.readlines()
    statefile.close()
    
    WaveFunction = FeedForwardNeuralNetwork(1, M)
    Hamiltonian = CalogeroSutherland(WaveFunction, 0.0, 0.0)
    
    plt.figure(figsize=(12,8))
    colors = cm.rainbow_r(np.linspace(0, 1, len(snapshots)))
    
    num_points = 200
    x = np.linspace(-5.0,5.0,num_points)
    
    for snapshot, color in zip(snapshots, colors):
        
        WaveFunction.alpha = np.array(snapshot.split()[1:]).astype(np.float)
        WaveFunction.separate()
        
        psi = np.zeros(num_points)
        for i in range(num_points):
            psi[i] = WaveFunction.calc_psi([x[i]])
            
        plt.plot(x, psi, color=color, linewidth=3, label=str(snapshot.split()[0]))
    
    psi_nonint = np.zeros(num_points)
    for i in range(num_points):
        psi_nonint[i] = Hamiltonian.nonint_gs_wavefunction([x[i]])
    plt.plot(x, psi_nonint, color='k', linewidth=2, linestyle='dashed', label=r'$\Psi_0^{non\text{-}int}(x)$')
    
    plt.ylim(-0.1,1.5)
    plt.xlim(-5,5)
    plt.ylabel(r'Ground state wave function $\Psi(x)$', fontsize=14)
    plt.xlabel(r'Position $x$ of one particle', fontsize=14)
    plt.title(r'Progression of supervised learning of initial parameters using '+str(M)+' hidden units', fontsize=16)
    plt.legend(loc='upper right', fontsize=14)
    plt.savefig('figures/N1_M'+str(M)+'_supervised_snapshots.pdf', format='pdf')

def plot_supervised_snapshots(N, M):
    
    if N == 1:
        plot_supervised_snapshots_N1(M)
        
    else:
        file = 'initialstates/N'+str(N)+'_M'+str(M)+'.txt'
        statefile = open(file, 'r')
        snapshots = statefile.readlines()
        statefile.close()
        
        WaveFunction = FeedForwardNeuralNetwork(N, M)
        Hamiltonian = CalogeroSutherland(WaveFunction, 0.0, 0.0)
        Sampler = ImportanceSampling(Hamiltonian, 0.001)
        
        plt.figure(figsize=(12,8))
        sns.set_style("whitegrid")
        colors = cm.rainbow_r(np.linspace(0, 1, len(snapshots)))
        num_samples = 10000
        num_skip = int(0.1*num_samples)
    
        for snapshot, color in zip(snapshots, colors):
            
            WaveFunction.alpha = np.array(snapshot.split()[1:]).astype(np.float)
            WaveFunction.separate()
            
            for sample in range(num_skip):
                accepted = Sampler.sample()
                
            x = np.empty(N*num_samples)
            for sample in range(num_samples):
                accepted = Sampler.sample()
                x[N*sample:N*(sample+1)] = WaveFunction.x
            sns.kdeplot(x, shade=True, color=color)
        
        x = np.random.normal(N*num_samples)
        sns.kdeplot(x, shade=False, color='k', linestyle='dashed')
        
        plt.ylim(-0.1,1.5)
        plt.xlim(-5,5)
        plt.ylabel(r'Probability distribution $|\Psi(x)|^2$', fontsize=14)
        plt.xlabel(r'Positions $x$ of '+str(N)+' particles', fontsize=14)
        plt.title(r'Progression of supervised learning of initial parameters for '+str(N)+' particles using '+str(M)+' hidden units', fontsize=16)
        plt.legend(loc='upper right')
        plt.savefig('figures/N'+str(N)+'_M'+str(M)+'_supervised_snapshots.pdf', format='pdf')

plot_supervised_snapshots(2,20)
