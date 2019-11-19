import numpy as np


class FeedForwardNeuralNetwork:

    def __init__(self, num_visible, num_hidden):
        """constructor"""
        self.N = num_visible                                     # number of particles
        self.M = num_hidden                                      # number of hidden units
        self.x = np.random.normal(0.0, 0.01, self.N)             # positions of particles
        self.b = np.random.normal(0.0, 0.1, self.M)              # bias
        self.w = np.random.normal(0.0, 0.1, self.M)              # weights for hidden to output
        self.W = np.random.normal(0.0, 0.1, (self.M,self.N))     # weights for visible to hidden
        self.alpha = self.vectorize(self.W, self.w, self.b)      # all parameters
        
    def supervised_train(self, X, y):
        """return mean square error for inputs X and targets y"""
        n = X.shape[0]                                           # number of data samples
            
            
    
        
    def calc_psi(self, x):
        """wave function"""
        h = np.dot(self.W,x) + self.b
        f = self.calc_f(h)
        return np.exp(np.dot(self.w,f))
        
        
    def calc_local_kinetic_energy(self, x):
        """local kinetic energy"""
        h = np.dot(self.W,x) + self.b
        ff = self.calc_ff(h)
        fff = self.calc_fff(h)
        
        KL = 0.0
        for p in range(self.N):
            foo = 0.0
            for i in range(self.M):
                KL += self.w[i]*self.W[i,p]**2*fff[i]
                foo += self.w[i]*self.W[i,p]*ff[i]
            KL += foo**2
        return -0.5*KL
    

    def calc_qforce(self, x, p):
        """quantum force on pth particle"""
        h = np.dot(self.W,x) + self.b
        ff = self.calc_ff(h)
        
        F = 0.0
        for i in range(self.M):
            F += self.w[i]*self.W[i,p]*ff[i]
        return 2.0*F
        
    
    def calc_gradient_logpsi(self, x):
        """gradient of logpsi with respect to all parameters"""
        h = np.dot(self.W,x) + self.b
        f = self.calc_f(h)
        ff = self.calc_ff(h)
        g = np.multiply(self.w,ff)
        gradient = self.vectorize(np.outer(g,x), g, f)
        return gradient
             
            
    def calc_f(self, z):
        """activation function"""
        return np.tanh(z)
    
    
    def calc_ff(self, z):
        """first derivative of activation function"""
        return 1.0-((np.tanh(z))**2)
    
    
    def calc_fff(self, z):
        """second derivative of activation function"""
        return 2.0*np.tanh(z)*(1.0-(np.tanh(z))**2)
        
        
    def vectorize(self, W, b, w):
        """return 1d array from matrix and vectors"""
        alpha = np.concatenate((np.ravel(W), b, w))
        return alpha
        
        
    def separate(self, alpha):
        """split parameter vector into weights and biases"""
        self.W = alpha[:self.M*self.N].reshape(self.M,self.N)
        self.b = alpha[self.M*self.N:self.M*self.N+self.M]
        self.w = alpha[-self.M:]
    
