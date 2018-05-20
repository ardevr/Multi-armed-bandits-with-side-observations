import numpy as np

class Result:
   
    def __init__(self, nbArms,Delta,T):
        self.nbArms = nbArms
        self.choices = np.zeros(T, dtype = int)
        self.obs     = (nbArms + 1)*np.ones(T,dtype = int)
        self.delta = Delta 
        self.regret = np.zeros(T)
    
    def store(self, t, choice):
        self.choices[t] = choice
        if t == 0 :
            self.regret[t] = self.delta[choice]
        else :
            self.regret[t] = self.regret[t-1] + self.delta[choice]
            
    def storeObs(self, t, choice):
        self.obs[t] = choice
    
    def getNbPulls(self):
        nbPulls = np.zeros(self.nbArms)
        for choice in self.choices:
            nbPulls[choice] += 1
        return nbPulls
    
    def getRegret(self):
        return self.regret
    
    def getPull(self):
        return self.choices
    
    def getNbObs(self):
        nbObs = np.zeros(self.nbArms)
        for obs in self.obs:
            if obs < self.nbArms :
                nbObs[obs] += 1
        return nbObs
        
    def getObs(self):
        return self.obs
    