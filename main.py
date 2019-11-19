import numpy as np
from wavefunction import FeedForwardNeuralNetwork
from hamiltonian import Calogero
from sampler import BruteForce, ImportanceSampling
from optimizer import StochasticGradientDescent, Adam
from vmc import VariationalMonteCarlo




def main():

    # model parameters
    num_particles = 1
    num_hidden = 20
    interaction_param = 0.0
    ramp_up_speed = 0.001
    max_step = 0.1
    #time_step = 0.001
    learning_rate = 0.01
    beta1 = 0.9
    beta2 = 0.999
    num_samples = 20000
    tolerance = 1E-6
    filename = "test.dat"

    # initialize objects
    WaveFunction = FeedForwardNeuralNetwork(num_particles,num_hidden)
    Hamiltonian = Calogero(WaveFunction, interaction_param, ramp_up_speed)
    Sampler = BruteForce(Hamiltonian, max_step)
    #Sampler = ImportanceSampling(Hamiltonian, time_step)
    Optimizer = Adam(WaveFunction, learning_rate, beta1, beta2)
    VMC = VariationalMonteCarlo(Optimizer, Sampler, Hamiltonian, num_samples, tolerance, filename)
    
    # run optimization
    VMC.minimize_energy()
    

    
    


if __name__ == "__main__":
    main()
