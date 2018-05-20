import numpy as np 
import random
from numpy.random import choice

class MOSS1():
    
    def __init__(self, nbArms, T):
        self.nbArms    = nbArms
        self.cumReward = np.zeros(self.nbArms)
        self.nbDraws   = np.zeros(self.nbArms)
        self.T         = T
        self.t         = 0
        
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
            a = np.log(self.T/(self.nbArms*self.nbDraws))
            UpperBound = self.cumReward/self.nbDraws+np.sqrt(a*(a>0)/self.nbDraws)
            OrdUpperBound = np.argsort(UpperBound)[::-1]
            It_choice = [OrdUpperBound[0]]
            for i in OrdUpperBound[1::]:
                if UpperBound[i] >= UpperBound[OrdUpperBound[0]]:
                    It_choice.append(i)
            It = choice(np.array(It_choice))
            It_choice.remove(int(It))
            if len(It_choice) == 0 :
                Jt = OrdUpperBound[1]
            else :
                Jt = choice(np.array(It_choice))
        self.t +=1
        return It,Jt
        
