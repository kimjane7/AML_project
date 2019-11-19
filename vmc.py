import os.path
import numpy as np

class VariationalMonteCarlo:

    def __init__(self, optimizer, sampler, hamiltonian, supervised_num_samples, reinforcement_num_samples, patience):
        """constructor"""
        
        self.optimizer = optimizer
        self.sampler = sampler
        self.hamiltonian = hamiltonian
        self.wavefunction = self.hamiltonian.wavefunction
        self.num_params = self.optimizer.num_params
        self.supervised_num_samples = supervised_num_samples
        self.reinforcement_num_samples = reinforcement_num_samples
        self.patience = patience
        
        # file names
        tag = 'N'+str(self.wavefunction.N)+'_M'+str(self.wavefunction.M)
        self.init_state_file = 'initialstates/'+tag+'_samples'+str(self.supervised_num_samples)+'_minMSE.txt'
        self.supervised_output_file = 'outputs/supervised_'+tag+'_samples'+str(self.supervised_num_samples)+'_minMSE.txt'
        self.reinforcement_output_file = 'outputs/reinforcement_'+tag+'_samples'+str(self.reinforcement_num_samples)+'.txt'
        
    

    def minimize_energy(self):
        """optimizes wave function parameters"""
        
        # if initial state file exists, load parameters
        # and continue to reinforcement learning part
        if os.path.isfile(self.init_state_file) and os.path.getsize(self.init_state_file) > 0:
            
            # skip over snapshots and extract optimized parameters
            statefile = open(self.init_state_file, 'r')
            init_state = statefile.readlines()[-1]
            statefile.close()
            
        # else do supervised learning of parameters for
        # non-interacting case and write to initial state file
        else:
            print('Supervised learning non-interacting case...')
            self.train_nonint_case()
        
        # train wave function for interacting case
        if self.hamiltonian.nu > 0.0:
            print('Reinforcement learning interacting case of nu = ', self.hamiltonian.nu, '...')
            self.optimizer.reset()
            self.train_int_case()


    def train_nonint_case(self):
        """supervised training of initial weights and biases
           of wave function using known wave function for
           non-interacting particles in 1D harmonic oscillator"""
           
        optimize = True
        min_cost = 100
        min_cost_cycle = 0
        cycles = 0
        
        # training progress
        outfile = open(self.supervised_output_file, 'w')
        outfile.write('{:<10s}{:<14s}{:<14s}{:<20s}\n'.format('# cycles', 'fidelity', 'MSE', '||MSE gradient||'))
        
        # snapshots and converged state
        statefile = open(self.init_state_file, 'w')
        snapshots = [0, 10, 100, 200, 500, 1000, 10000, 20000, 50000]
        iter = 0

        while optimize:
        
            # write snapshots
            if snapshots[iter] == cycles:
                statefile.write(str(cycles)+'\t'+str(self.wavefunction.alpha)+'\n')
                iter += 1
        
            # get data, calculate cost, update parameters
            X = np.random.normal(0.0, 1.0, (self.supervised_num_samples, self.wavefunction.N))
            self.calc_cost_gradient(X)
            self.optimizer.update_params(self.gradient)
            cycles += 1
            
            # write progress
            outfile.write('{:<10d}{:<14.8f}{:<14.8f}{:<20.8f}\n'.format(cycles, self.fidelity, self.cost, np.linalg.norm(self.gradient)))
            
            # stop training when cost doesn't decrease for 'patience' cycles
            if self.cost < min_cost:
                min_cost = self.cost
                min_cost_cycle = cycles
                
            elif abs(cycles-min_cost_cycle) == self.patience:
                optimize = False
                statefile.write(str(cycles)+'\t'+str(self.wavefunction.alpha)+'\n')
        
        outfile.close()
        statefile.close()



    def train_int_case(self):
        """reinforcement learning of ground state wave
           function in interacting case in which the
           interaction potential is gradually introduced"""
        
        optimize = True
        min_EL = 100
        min_EL_cycle = 0
        cycles = 0
        print('{:<10s}{:<20s}{:<20s}{:<20s}{:<20s}'.format('cycles', 'avg EL', 'var EL', '||gradient EL||', 'ratio accepted samples'))
        
        while optimize:
            
            self.estimate_gradient_local_energy()
            self.optimizer.update_params(self.gradient)
            cycles += 1
            
            print('{:<10d}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.15}'.format(cycles, self.avg_EL, self.var_EL, np.linalg.norm(self.gradient), self.ratio_accepted))
            
            # stop training when local energy doesn't decrease for 'patience' cycles
            if self.EL < min_EL:
                min_EL = self.EL
                min_EL_cycle = cycles
            elif abs(cycles-min_EL_cycle) == self.patience:
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
        for sample in range(self.supervised_num_samples):

            # calculate trial and target wave functions
            psi_trial = self.wavefunction.calc_psi(X[sample])
            psi_nonint = self.hamiltonian.nonint_gs_wavefunction(X[sample])
            
            # add up ratio of wave functions A for fidelity
            A = psi_nonint/(psi_trial+epsilon)
            avg_A += A
            avg_A2 += A**2
            
            # add up squared errors for MSE cost
            self.cost += (psi_trial-psi_nonint)**2
            
            # add up values for MSE gradient
            self.gradient += 2.0*(psi_trial-psi_nonint) \
                                  *self.wavefunction.calc_gradient_logpsi(X[sample])*psi_trial
    
        
        # calculate fidelity
        avg_A /= self.supervised_num_samples
        avg_A2 /= self.supervised_num_samples
        self.fidelity = avg_A**2/(avg_A2+epsilon)

        # calculate cost and its gradient
        self.cost /= self.supervised_num_samples
        self.gradient /= self.supervised_num_samples



    ''' this will be useful for fidelity training
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
        
    '''
            

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
            EL = self.hamiltonian.calc_local_energy(self.wavefunction.x, self.optimizer.t)
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
    
