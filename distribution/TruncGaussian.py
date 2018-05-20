from scipy.stats import truncnorm

class TruncGaussian():
    def __init__(self,mu=0,sigma=1,a=0,b=1):
        self.mu = mu
        self.sigma = sigma
        self.a = a #Lower bound of the distribution (0<=a<1)
        self.b = b # Upper bound of the distribution (0<b<=1)
    def draw(self):
        return self.mu + self.sigma*truncnorm.rvs(self.a,self.b,size=1)
    def expectation(self):
        return self.mu
