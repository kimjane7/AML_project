import numpy as np
from wavefunction import FeedForwardNeuralNetwork
from hamiltonian import CalogeroSutherland
from sampler import ImportanceSampling
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import rc
import seaborn as sns

plt.rcParams["mathtext.fontset"] = "dejavuserif"
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


def smooth_avg(x, y, n):

    num_points = int(np.ceil(len(x)/float(n)))
    X = np.zeros(num_points)
    Y = np.zeros(num_points)
    
    X[0] = x[0]
    Y[0] = y[0]
    
    for i in range(1,num_points):
        X[i] = x[int(i*n)]
        Y[i] = y[(i-1)*n:i*n].mean()
    X[-1] = x[-1]
    Y[-1] = y[(num_points-1)*n:].mean()
    
    return X, Y

##############################################################
##############################################################
##############################################################

def plot_reinforcement_EL(N, M, samples, nu):

    file = 'data/reinforcement_N'+str(N)+'_M'+str(M)+'_samples'+str(samples)+'_nu'+str(nu)+'.txt'
    data = np.loadtxt(file)
    
    cycle = data[:,0]
    EL = data[:,2]
    EL_var = data[:,3]
    E0 = 0.5*N+0.5*nu*N*(N-1)
    print("True ground state energy: E0 = {0}".format(E0))
    
    
    X, Y = smooth_avg(cycle, EL, 100)
    X, Y_var = smooth_avg(cycle, EL_var, 100)
    print("Final ground state energy prediction: E = {0}, variance = {1}".format(Y[-1], Y_var[-1]))
    
    fig, ax = plt.subplots(figsize=(12,8))
    plt.plot(cycle, EL, color='royalblue', alpha=0.4, linewidth=0.1, label='Raw Data')
    plt.plot(X, Y, color='royalblue', linewidth=2.0, label='Average Over 100 Iterations')
    plt.axhline(E0, color='k', linewidth=1.0, label='True Ground State Energy')
    plt.axhline(0.5*N, color='k', linewidth=1.0, linestyle='dashed', label='Non-Interacting Ground State Energy')
    ax.annotate('Final Energy Estimation = '+str(round(Y[-1],5)), xy=(0.999*cycle[-1], 0.99*E0), xytext=(0.5*cycle[-1], 0.5*(E0+0.5*N)), arrowprops=dict(facecolor='black', width=1, shrink=0.01, headwidth=5, headlength=5), size=20)
    
    
    plt.xlim(0, cycle[-1])
    plt.ylim(0,3*E0)
    plt.xlabel('Number of Iterations', fontsize=20)
    plt.ylabel(r'Average Local Energy $\langle E_L \rangle$', fontsize=20)
    plt.title(r'Ground State Energy Approximation for $N$ = '+str(N)+r'$, \ \nu$ = '+str(nu), fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(loc='upper right', fontsize=20)
    plt.savefig('figures/EL_N'+str(N)+'_M'+str(M)+'_samples'+str(samples)+'_nu'+str(nu)+'.pdf', format='pdf')

#plot_reinforcement_EL(2, 20, 10000, 2.0)
plot_reinforcement_EL(4, 20, 10000, 2.0)

##############################################################
##############################################################
##############################################################

def plot_reinforcement_fidelity(N, M, samples, nu):

    file = 'data/reinforcement_N'+str(N)+'_M'+str(M)+'_samples'+str(samples)+'_nu'+str(nu)+'.txt'
    data = np.loadtxt(file)
    
    cycle = data[:,0]
    fidelity = data[:,1]
    X, Y = smooth_avg(cycle, fidelity, 100)
    print("Final fidelity: F = {0}".format(Y[-1]))
    
    fig, ax = plt.subplots(figsize=(12,8))
    plt.plot(cycle, fidelity, color='royalblue', alpha=0.4, linewidth=0.1, label='Raw Data')
    plt.plot(X, Y, color='royalblue', linewidth=2.0, label='Average Over 100 Iterations')
    plt.axhline(fidelity[0], color='k', linestyle='dashed', linewidth=2.0, label='Initial Overlap Integral')
    ax.annotate('Final Overlap Integral = '+str(round(Y[-1],5)), xy=(0.999*cycle[-1], 0.99), xytext=(0.5*cycle[-1], 0.7), arrowprops=dict(facecolor='black', width=1, shrink=0.01, headwidth=5, headlength=5), size=20)
    
    plt.xlim(1, cycle[-1])
    plt.ylim(0,1)
    plt.xlabel('Number of Iterations', fontsize=20)
    plt.ylabel(r'Overlap Integral', fontsize=20)
    plt.title(r'Learning Curve for $N$ = '+str(N)+r'$, \ \nu$ = '+str(nu), fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(loc='lower right', fontsize=20)
    plt.savefig('figures/fidelity_N'+str(N)+'_M'+str(M)+'_samples'+str(samples)+'_nu'+str(nu)+'.pdf', format='pdf')

#plot_reinforcement_fidelity(2, 20, 10000, 2.0)

##############################################################
##############################################################
##############################################################


def plot_reinforcement_snapshots(N, M, samples, nu):

    file = 'states/reinforcement_N'+str(N)+'_M'+str(M)+'_samples'+str(samples)+'_nu'+str(nu)+'.txt'
    statefile = open(file, 'r')
    snapshots = statefile.readlines()
    statefile.close()
    
    WaveFunction = FeedForwardNeuralNetwork(N, M)
    Hamiltonian = CalogeroSutherland(WaveFunction, nu, 0.0)
    Sampler = ImportanceSampling(Hamiltonian, 0.001)
    
    plt.figure(figsize=(12,8))
    sns.set_style("whitegrid")
    colors = cm.rainbow_r(np.linspace(0, 1, len(snapshots)))
    num_samples = 300000
    num_skip = 5000

    index = [0, 10, 19, 29, 39]
    for snapshot, color in zip([snapshots[i] for i in index], [colors[i] for i in index]):
        
        WaveFunction.alpha = np.array(snapshot.split()[1:]).astype(np.float)
        WaveFunction.separate()
        WaveFunction.x = np.random.normal(0, 1/np.sqrt(2), N)
        
        for sample in range(num_skip):
            accepted = Sampler.sample()
            
        x = np.empty(N*num_samples)
        for sample in range(num_samples):
            accepted = Sampler.sample()
            x[N*sample:N*(sample+1)] = WaveFunction.x
            
        sns.kdeplot(x, shade=True, linewidth=2, color=color, label=str(snapshot.split()[0])+' updates')

     
    num_samples = 5000000
    WaveFunction.alpha = np.array(snapshots[-1].split()[1:]).astype(np.float)
    WaveFunction.separate()
    WaveFunction.x = np.random.normal(0, 1/np.sqrt(2), N)
    
    for sample in range(num_skip):
        accepted = Sampler.sample()
        
    x = np.empty(N*num_samples)
    for sample in range(num_samples):
        accepted = Sampler.sample()
        x[N*sample:N*(sample+1)] = WaveFunction.x
        
    sns.kdeplot(x, shade=True, linewidth=2, color=colors[-1], label=str(snapshots[-1].split()[0])+' updates')
    
    num_samples = 5000000
    x = np.empty(N*num_samples)
    WaveFunction.x = np.random.normal(0, 1/np.sqrt(2), N)
    
    for sample in range(num_skip):
        accepted = Sampler.exact_sample()
    
    for sample in range(num_samples):
        accepted = Sampler.exact_sample()
        x[N*sample:N*(sample+1)] = WaveFunction.x
    
    sns.kdeplot(x, shade=False, linewidth=3, color='k', linestyle='dashed', label=r'$|\Psi_0^{exact}(x)|^2$')
    

    plt.ylim(-0.1,0.7)
    plt.xlim(-4,4)
    plt.ylabel(r'Probability Distribution $|\Psi(x)|^2$', fontsize=20)
    plt.xlabel(r'Positions $x$', fontsize=20)
    plt.title('Reinforcement Learning of the Wave Function for $N$ = '+str(N)+r'$, \ \nu$ = '+str(nu), fontsize=24)
    plt.legend(loc='upper right', fontsize=20)
    plt.savefig('figures/N'+str(N)+'_M'+str(M)+'_reinforcement_snapshots.pdf', format='pdf')

#plot_reinforcement_snapshots(2, 20, 10000, 2.0)
plot_reinforcement_snapshots(4, 20, 10000, 2.0)

##############################################################
##############################################################
##############################################################

def plot_supervised_snapshots(N, M):
    
    if N == 1:
        file = 'states/supervised_N1_M'+str(M)+'.txt'
        statefile = open(file, 'r')
        snapshots = statefile.readlines()
        statefile.close()
        
        WaveFunction = FeedForwardNeuralNetwork(1, M)
        Hamiltonian = CalogeroSutherland(WaveFunction, 0.0, 0.0)
        
        fig, ax = plt.subplots(figsize=(12,8))
        colors = cm.rainbow_r(np.linspace(0, 1, len(snapshots)))
        
        num_points = 200
        x = np.linspace(-5.0,5.0,num_points)
        
        for snapshot, color in zip(snapshots, colors):
            
            WaveFunction.alpha = np.array(snapshot.split()[1:]).astype(np.float)
            WaveFunction.separate()
            
            psi = np.zeros(num_points)
            for i in range(num_points):
                psi[i] = WaveFunction.calc_psi([x[i]])
                
            plt.plot(x, psi, color=color, linewidth=3, label=str(snapshot.split()[0])+' updates')
        
        psi_nonint = np.zeros(num_points)
        for i in range(num_points):
            psi_nonint[i] = Hamiltonian.nonint_gs_wavefunction([x[i]])
        plt.plot(x, psi_nonint, color='k', linewidth=2, linestyle='dashed', label=r'$\Psi_0^{non\text{-}int}(x)$')
        
        plt.ylim(-0.1,1.5)
        plt.xlim(-5,5)
        plt.ylabel(r'Ground state wave function $\Psi(x)$', fontsize=20)
        plt.xlabel(r'Position $x$', fontsize=20)
        plt.title(r'Supervised learning of the wave function for the non-interacting case', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(loc='upper right', fontsize=20)
        plt.savefig('figures/N1_M'+str(M)+'_supervised_snapshots.pdf', format='pdf')
        
    else:
        file = 'states/supervised_N'+str(N)+'_M'+str(M)+'.txt'
        statefile = open(file, 'r')
        snapshots = statefile.readlines()
        statefile.close()
        
        WaveFunction = FeedForwardNeuralNetwork(N, M)
        Hamiltonian = CalogeroSutherland(WaveFunction, 0.0, 0.0)
        Sampler = ImportanceSampling(Hamiltonian, 0.005)
        
        fig, ax = plt.subplots(figsize=(12,8))
        sns.set_style("whitegrid")
        colors = cm.rainbow_r(np.linspace(0, 1, len(snapshots)))
        num_samples = 100000
        num_skip = 1
    
        
        #for snapshot, color in zip(snapshots, colors):
        index = [0, 2, 4, 5, 7, 9]
        for snapshot, color in zip([snapshots[i] for i in index], [colors[i] for i in index]):
            
            WaveFunction.alpha = np.array(snapshot.split()[1:]).astype(np.float)
            WaveFunction.separate()
            WaveFunction.x = np.random.normal(0, 1/np.sqrt(2), N)
            
            for sample in range(num_skip):
                accepted = Sampler.sample()
                
            x = np.empty(N*num_samples)
            for sample in range(num_samples):
                accepted = Sampler.sample()
                x[N*sample:N*(sample+1)] = WaveFunction.x
                
            sns.kdeplot(x, shade=True, linewidth=2, color=color, label=str(snapshot.split()[0])+' updates')
        
        num_samples = 1000000
        WaveFunction.alpha = np.array(snapshots[-1].split()[1:]).astype(np.float)
        WaveFunction.separate()
        WaveFunction.x = np.random.normal(0, 1/np.sqrt(2), N)
        
        for sample in range(num_skip):
            accepted = Sampler.sample()
            
        x = np.empty(N*num_samples)
        for sample in range(num_samples):
            accepted = Sampler.sample()
            x[N*sample:N*(sample+1)] = WaveFunction.x
            
        sns.kdeplot(x, shade=True, linewidth=2, color=colors[-1], label=str(snapshots[-1].split()[0])+' updates')
        
        num_samples = 10000000
        x = np.random.normal(0, 1/np.sqrt(2), N*num_samples)
        sns.kdeplot(x, shade=False, linewidth=3, color='k', linestyle='dashed', label=r'$|\Psi_0^{non\text{-}int}(x)|^2$')

        plt.ylim(-0.1,0.7)
        plt.xlim(-4,4)
        plt.ylabel(r'Probability Distribution $|\Psi(x)|^2$', fontsize=20)
        plt.xlabel(r'Positions $x$', fontsize=20)
        plt.title('Supervised Learning of the Wave Function for the Non-Interacting Case', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(loc='upper right', fontsize=20)
        plt.savefig('figures/N'+str(N)+'_M'+str(M)+'_supervised_snapshots.pdf', format='pdf')

#plot_supervised_snapshots(2, 20)
