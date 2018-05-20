# -*- coding: utf-8 -*-

__author__ = "Garcleon, Evrard"

import numpy as np 
from numpy.random import choice
class UCB_MultiObs():
    
    def __init__(self, nbArms):
        self.nbArms    = nbArms
        self.cumReward = np.zeros(self.nbArms)
        self.nbObs     = np.zeros(self.nbArms)
        self.t         = 0
        
    def startGame(self):
        self.t         = 0
        self.cumReward = np.zeros(self.nbArms)
        self.nbObs     = np.zeros(self.nbArms)

    def getReward(self, arm, reward,ObsType):
        self.cumReward[arm] += reward
        self.nbObs[arm]     +=1
                
    def choice(self):
        if self.t < self.nbArms:
            It = self.t
        else:
            UpperBound=(self.cumReward/self.nbObs)+np.sqrt(2*np.log(self.t)/self.nbObs)
            a = [np.argmax(UpperBound)]
            for i in range(self.nbArms) :
                if UpperBound[i] >= UpperBound[a[0]]:
                    a.append(i)
            It = choice(np.array(a))
        
        Jt = choice(np.array([i for i in range(self.nbArms) if i != It]),self.nbArms-1)
        self.t = self.t +1
        result = np.zeros(self.nbArms,dtype = int)
        result[0] = It
        result[1::] = Jt 
        return result

        
