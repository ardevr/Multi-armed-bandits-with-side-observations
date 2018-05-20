import numpy as np 
from numpy.random import choice

class UCB1_Double():
    
    def __init__(self, nbArms):
        self.nbArms    = nbArms
        self.cumReward = np.zeros(self.nbArms)
        self.nbObs     = np.zeros(self.nbArms)
        
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
            Jt = self.nbArms +1 
        else:
            UpperBound=(self.cumReward/self.nbObs)+np.sqrt(4*np.log(self.t)/self.nbObs)
            OrdUpperBound = np.argsort(UpperBound)
            a = [OrdUpperBound[-1]]
            for i in range(self.nbArms) :
                if UpperBound[i] >= UpperBound[a[0]]:
                    a.append(i)
            It = choice(np.array(a))
            b = a.remove(int(It))
            if b != None :
                Jt = choice(np.array(b))
            else :
                Jt = OrdUpperBound[-2]
        self.t = self.t +1 
        return (It,Jt)

        
