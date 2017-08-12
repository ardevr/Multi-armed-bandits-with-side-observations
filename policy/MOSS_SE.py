# -*- coding: utf-8 -*-
'''The MOSS-SE policy.
Reference = Bandits with Side Observations [GARCELON Evrard]'''

__author__ = "Garcelon, Evrard"

import numpy as np 
from numpy.random import choice

class MOSS_SE():
    
    def __init__(self, nbArms, T, alpha = 1/2, beta = 1/2):
        self.nbArms    = nbArms
        self.cumReward = np.zeros(self.nbArms)
        self.horizon   = T  
        self.arms      = list(range(self.nbArms))
        self.nbDraws   = np.zeros(self.nbArms)
        self.alpha     = alpha
        self.beta      = beta
        
    def startGame(self):
        self.t         = 0
        self.cumReward = np.zeros(self.nbArms)
        self.nbDraws   = np.zeros(self.nbArms)
        self.arms      = list(range(self.nbArms))

#Attention probleme elimine pas assez vite

    def armElimination(self):
        
        self.mean = self.cumReward/self.nbDraws
        self.a = self.alpha*max([np.log(self.horizon/self.t),1])
        self.Index1 = self.mean[self.arms] - np.sqrt(self.a/self.nbDraws[self.arms])
        self.Y = max(self.Index1)
        for i in self.arms :
            if (self.mean[i] + np.sqrt(self.a/self.nbDraws[i]) < self.Y) :
                self.arms.remove(i)

    
    def getReward(self, t, arm, reward):
        self.cumReward[arm]   += reward 
        self.nbDraws[arm]     += 1        
        self.t  = t 
        
    def getRewardDouble(self, t, arm1, arm2, reward1, reward2):
        self.cumReward[arm1]  += reward1       
        self.cumReward[arm2]  += reward2      
        self.nbDraws[arm1]    += 1
        self.nbDraws[arm2]    += 1
        self.t = t
       
    def choice(self):
        if self.t < self.nbArms:
            return self.t
        else:
            self.armElimination()
            self.b = np.log(self.horizon/(self.nbArms*self.nbDraws))
            self.mean = self.cumReward/self.nbDraws
            self.Index2 = self.mean+np.sqrt(self.b*(self.b>0)/self.nbDraws)
            for j in (np.argsort(self.Index2))[::-1]:
                if j in self.arms :
                    return j
                    break
                    
    def secondChoice(self):
        return int(choice(self.arms,1))
