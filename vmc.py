import time
import os.path
import numpy as np
import matplotlib.pyplot as plt

class VariationalMonteCarlo:

    def __init__(self, optimizer, sampler, num_samples):
        """constructor"""
        
        self.optimizer = optimizer
        self.sampler = sampler
        self.hamiltonian = self.sampler.hamiltonian
        self.wavefunction = self.hamiltonian.wavefunction
        self.num_params = self.optimizer.num_params
        self.num_samples = num_samples
        
        # file names
        tag = 'N'+str(self.wavefunction.N)+'_M'+str(self.wavefunction.M)
        
        self.supervised_state_file = 'states/supervised_'+tag+'.txt'
        self.supervised_data_file = 'data/supervised_'+tag+'.txt'
        
        self.reinforcement_state_file = 'states/reinforcement_'+tag+'_samples'+str(self.num_samples) \
        +'_nu'+str(self.hamiltonian.nu)+'.txt'
        self.reinforcement_data_file = 'data/reinforcement_'+tag+'_samples'+str(self.num_samples) \
                                        +'_nu'+str(self.hamiltonian.nu)+'.txt'
        
    
    def minimize_energy(self):
        """optimizes wave function parameters"""
        
        # if initial state file exists, load parameters
        # and continue to reinforcement learning part
        if os.path.isfile(self.supervised_state_file) and os.path.getsize(self.supervised_state_file) > 0:
            print('Reading initial state from file...')
            
            # skip over snapshots and extract optimized parameters
            statefile = open(self.supervised_state_file, 'r')
            init_state = statefile.readlines()[-1]
            statefile.close()
            
            # set initial parameters
            self.wavefunction.alpha = np.array(init_state.split()[1:]).astype(np.float)
            self.wavefunction.separate()
            
            
        # else do supervised learning of parameters for
        # non-interacting case and write to initial state file
        else:
            print('Supervised learning non-interacting case...')
            start = time.time()
            self.train_nonint_case()
            end = time.time()
            print('Done in {:.3f} seconds.'.format(end-start))
        
        # train wave function for interacting case
        if self.hamiltonian.nu > 0.0:
            print('Reinforcement learning interacting case of nu = ', self.hamiltonian.nu, '...')
            self.optimizer.reset()
            start = time.time()
            self.train_int_case()
            end = time.time()
            print('Done in {:.3f} seconds.'.format(end-start))


    def train_nonint_case(self):
        """supervised training of initial weights and biases
           of wave function using known wave function for
           non-interacting particles in 1D harmonic oscillator"""
           
        optimize = True
        cycles = 0
        patience = 50
        min_gradient = 1
        min_gradient_cycles = 0
        min_samples = 20000*self.wavefunction.N
        samples = min_samples

        # training progress
        datafile = open(self.supervised_data_file, 'w')
        datafile.write('{:<10s}{:<14s}{:<14s}{:<20s}\n'.format('# cycles', 'fidelity', 'MSE', '||MSE gradient||'))
        
        # snapshots of state
        statefile = open(self.supervised_state_file, 'w')
        iter = 0
        snapshots = [0, 10, 100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
        
        print('Starting with {0} samples...'.format(samples))

        while optimize:
        
            # write snapshots
            if snapshots[iter] == cycles:
                param_str = ''
                for param in self.wavefunction.alpha:
                    param_str += str(param)+' '
                statefile.write(str(cycles)+' '+param_str+'\n')
                iter += 1
                if iter == len(snapshots):
                    optimize = False
        
            # get data, calculate cost, update parameters
            X = np.concatenate((np.random.normal(0.0, 1.0, (int(0.9*samples), self.wavefunction.N)), \
                                np.random.uniform(-10.0, 10.0, (int(0.1*samples), self.wavefunction.N))))
            self.calc_cost_gradient(X)
            self.optimizer.update_params(self.gradient)
            cycles += 1
            
            # write progress
            datafile.write('{:<10d}{:<14.8f}{:<14.8f}{:<20.8f}\n'.format(cycles, self.fidelity, self.cost, np.linalg.norm(self.gradient)))
            
            # increase number of samples if gradient is too noisy
            if np.linalg.norm(self.gradient) < min_gradient:
                min_gradient = np.linalg.norm(self.gradient)
                min_gradient_cycles = cycles
                
            elif cycles-min_gradient_cycles > patience:
                samples += min_samples
                min_gradient = np.linalg.norm(self.gradient)
                min_gradient_cycles = cycles
                print('Increasing number of samples to {0} at cycle {1}...'.format(samples, cycles))
        
        datafile.close()
        statefile.close()



    def train_int_case(self):
        """reinforcement learning of ground state wave
           function in interacting case in which the
           interaction potential is gradually introduced"""
        
        optimize = True
        min_EL = 100
        min_EL_cycle = 0
        cycles = 0
        
        # training progress
        datafile = open(self.reinforcement_data_file, 'w')
        datafile.write('{:<10s}{:<20s}{:<20s}{:<20s}{:<20s}{:<20s}\n'.format('# cycles', 'fidelity', 'EL avg', 'EL var', '||MSE gradient||', 'ratio accepted samples'))
        
        # snapshots of state
        statefile = open(self.reinforcement_state_file, 'w')
        iter = 0
        snapshots = [1000*i for i in range(61)]
        
        print('Using {0} samples...'.format(self.num_samples))
        
        while optimize:
        
            # write snapshots
            if snapshots[iter] == cycles:
                print('Writing snapshot to file at cycle {0}...'.format(cycles))
                param_str = ''
                for param in self.wavefunction.alpha:
                    param_str += str(param)+' '
                statefile.write(str(cycles)+' '+param_str+'\n')
                iter += 1
                if iter == len(snapshots):
                    optimize = False
            
            # calculate energy, update parameters
            self.calc_local_energy_gradient()
            self.optimizer.update_params(self.gradient)
            cycles += 1
            
            # write progress
            datafile.write('{:<10d}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.5f}{:<20.15}\n'.format(cycles, self.fidelity, self.avg_EL, self.var_EL, np.linalg.norm(self.gradient), self.ratio_accepted))
        
        datafile.close()
        statefile.close()


    def calc_cost_gradient(self, X):
        """calculate MSE, MSE gradient, and fidelity"""
        
        self.cost = 0.0
        self.gradient = 0.0
        avg_A = 0.0
        avg_A2 = 0.0
        self.fidelity = 0.0
        samples = X.shape[0]

        # prevent division by zero
        epsilon = 1e-8
    
        # take samples
        for sample in range(samples):

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
        self.fidelity = avg_A**2/(samples*avg_A2+epsilon)

        # calculate cost and its gradient
        self.cost /= samples
        self.gradient /= samples
        
            

    def calc_local_energy_gradient(self):
        """estimate gradient of average local energy
           with respect to wave function parameters"""

        num_accepted = 0
        avg_A = 0.0
        avg_A2 = 0.0
        self.fidelity = 0.0
        self.avg_EL = 0.0
        avg_EL2 = 0.0
        avg_gradient_logpsi = np.zeros(self.num_params)
        avg_EL_gradient_logpsi = np.zeros(self.num_params)
        
        # prevent division by zero
        epsilon = 1e-8
        
        
        # let sampler to reach equilibrium
        self.wavefunction.x = np.random.normal(0, 0.1, self.wavefunction.N)
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
            
            # calculate trial and target wave functions
            psi_trial = self.wavefunction.calc_psi(self.wavefunction.x)
            psi_exact = self.hamiltonian.exact_gs_wavefunction(self.wavefunction.x)
            
            # add up ratio of wave functions A for fidelity
            A = psi_exact/(psi_trial+epsilon)
            avg_A += A
            avg_A2 += A**2
            
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
        
        # calculate fidelity
        self.fidelity = avg_A**2/(num_effective_samples*avg_A2+epsilon)
        
        # calculate gradient of local energy
        self.gradient = 2.0*(avg_EL_gradient_logpsi-self.avg_EL*avg_gradient_logpsi)
    
