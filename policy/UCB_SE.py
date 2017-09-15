# -*- coding: utf-8 -*-
'''The UCB-SE policy.
Reference = Bandits with Side Observations [GARCELON Evrard]'''

__author__ = "Garcleon, Evrard"

import numpy as np 
from numpy.random import choice

class UCB_SE():
    
    def __init__(self, nbArms, horizon, alpha = 1/2, beta = 3/2):
        self.nbArms    = nbArms
        self.horizon   = horizon
        self.alpha     = alpha
        self.beta      = beta
        self.cumReward = np.zeros(self.nbArms)
        self.nbDraws   = np.zeros(self.nbArms)
        self.arms      = list(range(self.nbArms))
    
    def startGame(self):
        self.t         = 0
        self.cumReward = np.zeros(self.nbArms)
        self.nbDraws   = np.zeros(self.nbArms)
        self.arms      = list(range(self.nbArms))
        
    def armElimination(self):
        self.mean   = self.cumReward/self.nbDraws
        self.Index1 = self.mean[self.arms] - np.sqrt(self.alpha*max(np.log(self.horizon/self.t),1)/self.nbDraws[self.arms])
        self.Y      = max(self.Index1)
        for i in self.arms :
            if self.mean[i] + np.sqrt(self.alpha*max(np.log(self.horizon/self.t),1)/self.nbDraws[i]) <= self.Y :
                self.arms.remove(i)

        
    def getReward(self, t, arm, reward):
        self.cumReward[arm] += reward
        self.nbDraws[arm]   +=1
        self.t               = t+1
            
    def choice(self):
        if self.t < self.nbArms:
            return self.t
        else:
            self.armElimination()
            self.Index2 = (self.cumReward/self.nbDraws)[self.arms] + np.sqrt(self.beta*np.log(self.t)/self.nbDraws[self.arms])
            return np.argmax(self.Index2)
    
    def secondChoice(self):
        return int(choice(self.arms,1))
        
        