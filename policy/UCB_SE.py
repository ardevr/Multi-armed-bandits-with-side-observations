
import numpy as np 
from numpy.random import choice

class UCB_SE():
    
    def __init__(self, nbArms, alpha = 2, beta = 3/2):
        self.nbArms    = nbArms
        self.alpha     = alpha
        self.beta      = beta
        self.cumReward = np.zeros(self.nbArms)
        self.nbDraws   = np.zeros(self.nbArms)
        self.arms      = list(range(self.nbArms))

    def startGame(self):
        self.t         = 0
        self.nbReward  = 0
        self.epoch     = self.nbArms + 1
        self.epsilon   = []
        self.cumReward = np.zeros(self.nbArms)
        self.nbDraws   = np.zeros(self.nbArms)
        self.arms      = list(range(self.nbArms))
    
    def armElimination(self):
        mean   = self.cumReward/self.nbDraws
        Index1 = mean[self.arms] - np.sqrt(self.alpha*max(np.log(self.epoch/self.t),1)/self.nbDraws[self.arms])
        Y      = max(Index1)
        for i in self.arms :
            if mean[i] + np.sqrt(self.alpha*max(np.log(self.epoch/self.t),1)/self.nbDraws[i]) < Y :
                self.arms.remove(i)

    def getReward(self, arm, reward):
        self.cumReward[arm] += reward
        self.nbDraws[arm]   +=1
        if self.t >= self.nbArms:
            self.nbReward = self.nbReward + 1
        if self.nbReward == 2 :
            (self.epsilon).append(1)
            self.nbReward = 0
    
    def choice(self):
        if self.t < self.nbArms :
            It,Jt = self.t, self.nbArms + 1
        else:
            self.armElimination()
            if self.nbReward == 1:
                self.epsilon.append(0)
                self.nbReward = 0
            index = (self.cumReward/self.nbDraws) + np.sqrt(self.beta*max(np.log(self.epoch/self.t),1)/self.nbDraws)
            gamma = min(index)
            for j in range(self.nbArms) :
                if not (j in self.arms) :
                    index[j] = gamma -1
            a = np.argsort(index)[::-1]
            b = [a[0]]
            for j in a[1::] :
                if index[j] >= index[b[0]]:
                    b.append(j)
            It = choice(np.array(b))
            Jt = choice(self.arms)
            if self.t == self.epoch :
                mean_epsilon = np.mean(np.array(self.epsilon))
                if mean_epsilon > 0 :
                    if len(self.arms) == 1 and self.t >= 8*np.log(16/mean_epsilon)/mean_epsilon :
                        self.epoch =  int(np.exp(mean_epsilon*self.epoch/8))
                    else :
                        self.epoch = 2*self.epoch
                        self.arms = list(range(self.nbArms))
                
                else :
                    self.epoch = 2*self.epoch
                    self.t = -1
                    self.arms = list(range(self.nbArms))
                    self.cumReward = np.zeros(self.nbArms)
                    self.nbDraws = np.zeros(self.nbArms)
        self.t = self.t + 1        
        return It,Jt
        
        
        