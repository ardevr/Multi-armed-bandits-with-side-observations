import numpy as np 
import random
from numpy.random import choice

class UCB_PropObs():
    
    def __init__(self, nbArms, Delta):
        self.nbArms    = nbArms
        self.cumReward = np.zeros(self.nbArms)
        self.nbDraws   = np.zeros(self.nbArms)
        self.Delta = Delta
    
        
    def startGame(self):
        self.t         = 0
        self.cumReward = np.zeros(self.nbArms)
        self.nbDraws   = np.zeros(self.nbArms)

    def getRewardPull(self, arm, reward):
        self.cumReward[arm] +=reward
        self.nbDraws[arm]   +=1 
        
    def getRewardObs(self,arm,reward):
        self.getRewardPull(arm,reward)
    
    def choice(self):
        if self.t < self.nbArms :
            It = self.t
            Jt = self.nbArms + 1
        else :
            UpperBound = self.cumReward/self.nbDraws + np.sqrt(4*np.log(self.t)/self.nbDraws)
            It_choice = [np.argmax(UpperBound)]
            for i in range(self.nbArms) :
                if UpperBound[i] >= UpperBound[It_choice[0]]:
                    It_choice.append(i)
            It = choice(np.array(It_choice))
            proba = 1/self.Delta
            proba[0] = 0
            Jt = choice(np.linspace(0,self.nbArms-1,self.nbArms,dtype = int), p = proba/np.sum(proba))
            if Jt == It :
                Jt = self.nbArms+1 
        self.t +=1
        return It,Jt
        
