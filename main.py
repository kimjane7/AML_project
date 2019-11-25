import numpy as np
from wavefunction import FeedForwardNeuralNetwork
from hamiltonian import CalogeroSutherland
from sampler import ImportanceSampling
from optimizer import Adam
from vmc import VariationalMonteCarlo




def main():

    # model parameters
    num_particles = 2
    num_hidden = 20
    interaction_param = 0.5
    ramp_up_speed = 0.001
    time_step = 0.001
    num_samples = 500000


    # initialize objects
    WaveFunction = FeedForwardNeuralNetwork(num_particles, num_hidden)
    Hamiltonian = CalogeroSutherland(WaveFunction, interaction_param, ramp_up_speed)
    Sampler = ImportanceSampling(Hamiltonian, time_step)
    Optimizer = Adam(WaveFunction)
    VMC = VariationalMonteCarlo(Optimizer, Sampler, num_samples)
    
    
    # run optimization
    VMC.minimize_energy()
    

    
    


if __name__ == "__main__":
    main()
