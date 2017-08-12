# -*- coding: utf-8 -*-
'''The UCB-SE policy.
Reference = Bandits with Side Observations [GARCELON Evrard]'''

__author__ = "Garcleon, Evrard"

import numpy as np 
from numpy.random import choice

class UCB_SE_Pot():
    
    def __init__(self, nbArms, horizon, psi = lambda x : np.exp(x), alpha = 1/2, beta = 1/2):
        self.nbArms = nbArms
        self.horizon = horizon
        self.alpha = alpha
        self.psi = psi
        self.beta = beta
        self.cumReward = np.zeros(self.nbArms)
        self.nbDraws = np.zeros(self.nbArms)
        self.arms = list(range(self.nbArms))
    
    def startGame(self):
        self.t = 0
        self.cumReward = np.zeros(self.nbArms)
        self.nbDraws = np.zeros(self.nbArms)
        self.arms = list(range(self.nbArms))
        
    def armElimination(self):
        self.mean = self.cumReward/self.nbDraws
        self.Index1 = self.mean[self.arms] - np.sqrt(self.alpha*max(np.log(self.horizon/self.t),1)/self.nbDraws[self.arms])
        self.Y = max(self.Index1)
        for i in self.arms:
            if self.mean[i] + np.sqrt(self.alpha*max(np.log(self.horizon/self.t),1)/self.nbDraws[i]) <= self.Y :
                self.arms.remove(i)
        
    def getReward(self, t, arm, reward):
        self.cumReward[arm] += reward
        self.nbDraws[arm] +=1
        self.t = t
    
    def getRewardDouble(self, t, arm1, arm2, reward1, reward2):
        self.cumReward[arm1] += reward1
        self.cumReward[arm2] += reward2
        self.nbDraws[arm1]   += 1
        self.nbDraws[arm2]   += 1
        self.t = t
        
    def choice(self):
        if self.t < self.nbArms:
            return self.t
        else:
            self.armElimination()
            self.mean = self.cumReward/self.nbDraws
            self.Index2 = self.mean+ np.sqrt(self.beta*np.log(self.t)/self.nbDraws)
            for j in (np.argsort(self.Index2))[::-1]:
                if j in self.arms :
                    return j
                    break
    
    def secondChoice(self):
        self.P = 1/self.psi(self.cumReward/self.nbDraws)[self.arms]
        self.c = np.sum(self.P)
        self.p = (self.P/self.c)
        return int(choice(self.arms,1,p = self.p))
        
        