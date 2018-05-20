import numpy as np

class Evaluation():
  
    def __init__(self, env, pol, nbRepetitions, T, epsilon, Delta, tsav=[]):
        if len(tsav)>0:
            self.tsav = tsav
        else:
            self.tsav = np.arange(T)
        self.env = env
        self.pol = pol
        self.nbRepetitions = nbRepetitions
        self.T = T
        self.nbArms = env.nbArms
        self.epsilon = epsilon
        self.nbPulls = np.zeros((self.nbRepetitions, self.nbArms))
        self.nbObs = np.zeros((self.nbRepetitions, self.nbArms))
        self.regret = np.zeros((self.nbRepetitions, len(tsav)))
        self.obs = np.zeros((self.nbRepetitions, T))
        self.pull = np.zeros((self.nbRepetitions, T))
        for k in range(nbRepetitions) :
            result = env.play(self.pol, self.epsilon, self.T)
            self.nbPulls[k,:] = result.getNbPulls()
            self.regret[k,:] = result.regret[self.tsav]
            self.nbObs[k,:] = result.getNbObs()
            self.obs[k,:] = result.getObs()
            self.pull[k,:] = result.getPull()

    def meanNbDraws(self):
        return np.mean(self.nbPulls ,axis = 0) 

    def meanRegret(self):
        return np.mean(self.regret,axis=0)
    
    def quantile(self,q):
        return np.percentile(self.regret, q, axis=0)
    
    def meanObs(self):
        return np.mean(self.nbObs,axis = 0)
    
    def ratio(self) :
        ratio  = np.zeros((self.nbArms,self.T))
        cumObs = np.zeros((self.nbRepetitions,self.nbArms,self.T))
        cumPull= np.zeros((self.nbRepetitions,self.nbArms,self.T))
        for i in range(self.nbArms) :
            for k in range(self.nbRepetitions):
                cumObs[k,i,:] = np.cumsum(self.obs[k,:] == i)
                cumPull[k,i,:] = np.cumsum(self.pull[k,:] == i)
        cumObs = np.mean(cumObs,axis = 0)
        cumPull = np.mean(cumPull,axis = 0) 
        ratio = cumObs/(cumObs+cumPull)
        return ratio  
            
