from wavefunction import FeedForwardNeuralNetwork
from hamiltonian import Calogero
from sampler import BruteForce
from optimizer import StochasticGradientDescent
from vmc import VariationalMonteCarlo
import numpy as np



def main():

    # model parameters
    num_particles = 1
    num_hidden = 4
    interaction_param = 0.0
    max_step = 0.5
    num_samples = 100000
    tolerance = 1E-7
    filename = "test.dat"

    # initialize objects
    WaveFunction = FeedForwardNeuralNetwork(num_particles,num_hidden)
    Hamiltonian = Calogero(WaveFunction, interaction_param)
    Sampler = BruteForce(Hamiltonian, max_step)
    Optimizer = StochasticGradientDescent(WaveFunction)
    VMC = VariationalMonteCarlo(Optimizer, Sampler, Hamiltonian, num_samples, tolerance, filename)
    
    """
    # plot samples drawn from exact wave function
    plotfile = "exactdistribution_N"+str(num_particles)+"_nu"+str(interaction_param)\
               +"_samples"+str(num_samples)+".pdf"
    Sampler.plot_samples(num_samples, plotfile, use_exact=True)
    """
    
    # run optimization
    VMC.minimize_energy()
    
    


if __name__ == "__main__":
    main()
