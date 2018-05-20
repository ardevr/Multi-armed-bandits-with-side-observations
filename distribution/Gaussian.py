from scipy.stats import norm

class Gaussian():
    def __init__(self,mu=0,sigma=1):
        self.mu = mu
        self.sigma = sigma
    def draw(self):
        return self.mu + self.sigma*norm.rvs()
    def expectation(self):
        return self.mu
