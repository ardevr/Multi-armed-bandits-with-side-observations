# -*- coding: utf-8 -*-
'''The UCB1-Uniforme policy'''

__author__ = "Garcleon, Evrard"

import numpy as np 
from numpy.random import choice

class FTL():
    
    def __init__(self, nbArms):
        self.nbArms    = nbArms
        self.cumReward = np.zeros(self.nbArms)
        self.nbObs     = np.zeros(self.nbArms)
        self.t         = 0
        
    def startGame(self):
        self.t         = 0
        self.cumReward = np.zeros(self.nbArms)
        self.nbObs     = np.zeros(self.nbArms)

    def getReward(self, arm, reward,typeobs):
        self.cumReward[arm] += reward
        self.nbObs[arm]     +=1
                
    def choice(self):
        if self.t < self.nbArms:
            It = self.t
        else:
            UpperBound = self.cumReward/self.nbObs
            a = [np.argmax(UpperBound)]
            for i in range(self.nbArms) :
                if UpperBound[i] >= UpperBound[a[0]]:
                    a.append(i)
            It = choice(np.array(a))
        Jt = 1-It
        self.t = self.t +1 
        return (It,Jt)

        
