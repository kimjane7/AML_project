from wavefunction import FeedForwardNeuralNetwork
from hamiltonian import Calogero
from metropolis import BruteForce
from sgd import StochasticGradientDescent
from vmc import VariationalMonteCarlo
import numpy as np


def main():

    # model parameters
    num_particles = 2
    num_hidden = 3
    interaction_param = 2.0
    max_step = 0.3
    learning_rate = 0.01
    num_samples = 1000
    filename = "test.dat"

    # objects
    WaveFunction = FeedForwardNeuralNetwork(num_particles,num_hidden)
    Hamiltonian = Calogero(WaveFunction, interaction_param)
    Sampler = BruteForce(WaveFunction, max_step)
    Optimizer = StochasticGradientDescent(WaveFunction, learning_rate)
    VMC = VariationalMonteCarlo(Optimizer, Sampler, Hamiltonian, num_samples, filename)




if __name__ == "__main__":
    main()
