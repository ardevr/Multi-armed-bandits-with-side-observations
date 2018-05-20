import numpy as np 
from numpy.random import choice

class UCB1_Uniforme():
    
    def __init__(self, nbArms):
        self.nbArms    = nbArms
        self.cumReward = np.zeros(self.nbArms)
        self.nbObs     = np.zeros(self.nbArms)
        self.t         = 0
        
    def startGame(self):
        self.t         = 0
        self.cumReward = np.zeros(self.nbArms)
        self.nbObs     = np.zeros(self.nbArms)

    def getRewardPull(self, arm, reward):
        self.cumReward[arm] += reward
        self.nbObs[arm]     +=1
    
    def getRewardObs(self,arm,reward):
        self.getRewardPull(arm,reward)
                
    def choice(self):
        if self.t < self.nbArms:
            It = self.t
        else:
            UpperBound=(self.cumReward/self.nbObs)+np.sqrt(4*np.log(self.t)/self.nbObs)
            a = [np.argmax(UpperBound)]
            for i in range(self.nbArms) :
                if UpperBound[i] >= UpperBound[a[0]]:
                    a.append(i)
            It = choice(np.array(a))
        Jt = choice(list(range(self.nbArms)))
        if Jt == It :
            Jt = self.nbArms+1
        self.t = self.t +1 
        return (It,Jt)

        
