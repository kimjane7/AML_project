import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
        """optimizes wave function parameters"""
        
        # get weights and biases for non-interacting case
        print('='*73)
        print('Supervised learning of the non-interacting case...')
        print('='*73)
        self.train_nonint_case()
        print('Done.')
        
        
        """CHECK"""
        '''
        # train wave function for interacting case
        if self.hamiltonian.nu > 0.0:
            print('Reinforcement learning of the interacting case of nu = ', self.hamiltonian.nu, '...')
            self.train_int_case()
            print('Done.')
        '''


    def train_nonint_case(self):
        """supervised training of initial weights and biases
           of wave function using known wave function for
           non-interacting particles in 1D harmonic oscillator"""
           
        optimize = True
        cycles = 0
        print('{:<10s}{:<14s}{:<14s}{:<20s}'.format('cycles', 'fidelity', 'MSE', '||MSE gradient||'))
        
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
        
        while optimize:
        
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
                break
                
                
            X = np.random.normal(0.0, 1.0, (self.num_samples, self.wavefunction.N))
            self.calc_cost_gradient(X)
            self.optimizer.update_params(self.gradient)
            cycles += 1
            
            print('{:<10d}{:<14.5f}{:<14.5f}{:<20.5f}'.format(cycles, self.fidelity, self.cost, np.linalg.norm(self.gradient)))
            
            if np.linalg.norm(self.gradient) < self.tolerance:
                optimize = False




    """CHECK"""
    def train_int_case(self):
        """reinforcement learning of ground state wave
           function in interacting case in which the
           interaction potential is gradually introduced"""
        
        optimize = True
        cycles = 0
        print('{:<10s}{:<20s}{:<20s}{:<20s}{:<20s}'.format('cycles', 'avg EL', 'var EL', '||gradient EL||', 'ratio accepted samples'))
        
        while optimize:
            
            self.estimate_gradient_local_energy()
            self.optimizer.update_params(self.gradient)
            cycles += 1
            
            print('{:<10d}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.15}'.format(cycles, self.avg_EL, self.var_EL, np.linalg.norm(self.gradient), self.ratio_accepted))
            
            if np.linalg.norm(self.gradient) < self.tolerance:
                optimize = False



    def calc_cost_gradient(self, X):
        """calculate MSE, MSE gradient, and fidelity"""
        
        self.cost = 0.0
        self.gradient = 0.0
        avg_A = 0.0
        avg_A2 = 0.0
        self.fidelity = 0.0

        # prevent division by zero
        epsilon = 1e-8
    
        # take samples
        for sample in range(self.num_samples):
            
            # draw positions from non-interacting ground state (normal distribution)
            #x = np.random.normal(0.0, 1.0, self.wavefunction.N)
            #x = np.random.uniform(-5.0, 5.0, self.wavefunction.N)

            # calculate trial and target wave functions
            psi_trial = self.wavefunction.calc_psi(X[sample])
            psi_nonint = self.hamiltonian.nonint_gs_wavefunction(X[sample])
            
            #print("trial", psi_trial)
            #print("nonint", psi_nonint)
        
            
            # add up ratio of wave functions A for fidelity
            A = psi_nonint/(psi_trial+epsilon)
            avg_A += A
            avg_A2 += A**2
            
            # add up squared errors for MSE cost
            self.cost += (psi_trial-psi_nonint)**2
            
            # add up values for MSE gradient
            #print("x = {0}, grad logpsi = {1}".format(x, self.wavefunction.calc_gradient_logpsi(x)))
            self.gradient += 2.0*(psi_trial-psi_nonint) \
                                  *self.wavefunction.calc_gradient_logpsi(X[sample])*psi_trial
    
        
        # calculate fidelity
        avg_A /= self.num_samples
        avg_A2 /= self.num_samples
        self.fidelity = avg_A**2/(avg_A2+epsilon)


        # calculate cost and its gradient
        self.cost /= self.num_samples
        self.gradient /= self.num_samples
    
        """
        plt.figure(figsize=(8,6))
        plt.rc('font', family='serif')
        plt.rc('text', usetex=True)
        sns.set_style("whitegrid")
        sns.kdeplot(np.ravel(X), shade=True)
        plt.show()
        """


    def estimate_gradient_nonint_overlap(self):
        """estimate gradient of overlap integral between trial
           wave function and non-interacting ground state"""
           
        num_accepted = 0
        avg_A = 0.0
        avg_A2 = 0.0
        self.fidelity = 0.0

        # let sampler to reach equilibrium
        fraction_skip = 0.2
        num_skip_samples = int(fraction_skip*self.num_samples)
        num_effective_samples = self.num_samples-num_skip_samples
        for sample in range(num_skip_samples):
            accepted = self.sampler.sample()
            
        # for debugging
        avg_position = 0.0
        
        # prevent division by zero
        epsilon = 1e-8
        
        # take samples to estimate gradient of overlap
        for sample in range(num_effective_samples):
            
            # count accepted samples
            accepted = self.sampler.sample()
            
            avg_position += self.wavefunction.x[0]
            
            if accepted:
                num_accepted += 1
            
            # calculate ratio of wave functions A and gradient of trial wave function
            A = self.hamiltonian.nonint_gs_wavefunction(self.wavefunction.x)/ \
                (self.wavefunction.calc_psi(self.wavefunction.x)+epsilon)
            #print("top = ", self.hamiltonian.nonint_gs_wavefunction(self.wavefunction.x))
            #print("bottom = ", self.wavefunction.calc_psi(self.wavefunction.x))
            #print("position = ", self.wavefunction.x)
            gradient_logpsi = self.wavefunction.calc_gradient_logpsi(self.wavefunction.x)
            
            # add up values for expectation values
            avg_A += A
            avg_A2 += A**2
            avg_gradient_logpsi += gradient_logpsi
            avg_A_gradient_logpsi += A*gradient_logpsi
        
        # calculate expectivation values
        avg_A /= num_effective_samples
        avg_A2 /= num_effective_samples
        avg_gradient_logpsi /= num_effective_samples
        avg_A_gradient_logpsi /= num_effective_samples
        
        # calculate ratio accepted
        self.ratio_accepted = float(num_accepted)/num_effective_samples
        
        # calculate overlap integral and its gradient
        self.overlap = avg_A**2/(avg_A2+epsilon)
        self.gradient = 2.0*self.overlap*(avg_A_gradient_logpsi/(avg_A+epsilon)-avg_gradient_logpsi)
        
        print("avg position = ", avg_position/num_effective_samples)
        
            

    def estimate_gradient_local_energy(self):
        """estimate gradient of average local energy
           with respect to wave function parameters"""

        num_accepted = 0
        self.avg_EL = 0.0
        avg_EL2 = 0.0
        avg_gradient_logpsi = np.zeros(self.num_params)
        avg_EL_gradient_logpsi = np.zeros(self.num_params)
        
        # let sampler to reach equilibrium
        fraction_skip = 0.2
        num_skip_samples = int(fraction_skip*self.num_samples)
        num_effective_samples = self.num_samples-num_skip_samples
        for sample in range(num_skip_samples):
            accepted = self.sampler.sample()
            
        # take samples to estimate gradient of local energy
        for sample in range(num_effective_samples):
        
            # count accepted samples
            accepted = self.sampler.sample()
            if accepted:
                num_accepted += 1
            
            # calculate local energy and gradient of trial wave function
            EL = self.hamiltonian.calc_local_energy(self.wavefunction.x,1000)
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
        
        # calculate gradient of local energy
        self.gradient = 2.0*(avg_EL_gradient_logpsi-self.avg_EL*avg_gradient_logpsi)
    
