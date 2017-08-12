# -*- coding: utf-8 -*-
'''The MOSS1 policy.
Reference = Bandits with Side Observations [GARCELON Evrard]'''

__author__ = "Garcleon, Evrard"

import numpy as np 
import random

class MOSS1():
    
    def __init__(self, nbArms, T):
        self.nbArms    = nbArms
        self.cumReward = np.zeros(self.nbArms)
        self.nbDraws   = np.zeros(self.nbArms)
        self.T         = T 
        
    def startGame(self):
        self.t         = 0
        self.cumReward = np.zeros(self.nbArms)
        self.nbDraws   = np.zeros(self.nbArms)

    def getReward(self, t, arm, reward):
        self.cumReward[arm] +=reward
        self.nbDraws[arm]   +=1 
        self.t               =t 
        
    def getRewardDouble(self, t, arm1, arm2, reward1, reward2):
        self.cumReward[arm1] += reward1
        self.cumReward[arm2] += reward2
        self.nbDraws[arm1]   += 1
        self.nbDraws[arm2]   += 1
        self.t               =t
    
    def choice(self):
        if self.t < self.nbArms:
            return self.t
        else:
            self.a = np.log(self.T/(self.nbArms*self.nbDraws))
            return np.argsort(self.cumReward/self.nbDraws+np.sqrt(self.a*(self.a>0)/self.nbDraws))[-1]
        
    def secondChoice(self):
        self.a = np.log(self.T/(self.nbArms*self.nbDraws))
        return np.argsort(self.cumReward/self.nbDraws+np.sqrt(self.a*(self.a>0)/self.nbDraws))[-2]
        
