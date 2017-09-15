# -*- coding: utf-8 -*-
'''The MOSS1 policy.'''

__author__ = "Garcleon, Evrard"

import numpy as np 
import random

class MOSS1():
    
    def __init__(self, nbArms, T):
        self.nbArms    = nbArms
        self.cumReward = np.zeros(self.nbArms)
        self.nbDraws   = np.zeros(self.nbArms)
        self.T         = T
        self.a         = 0 
        
    def startGame(self):
        self.t         = 0
        self.cumReward = np.zeros(self.nbArms)
        self.nbDraws   = np.zeros(self.nbArms)
        self.a         = 0

    def getReward(self, t, arm, reward):
        self.cumReward[arm] +=reward
        self.nbDraws[arm]   +=1 
        self.t               =t+1
        
    
    def choice(self):
        if self.t < self.nbArms:
            return self.t
        else:
            self.a = np.log(self.T/(self.nbArms*self.nbDraws))
            return np.argmax(self.cumReward/self.nbDraws+np.sqrt(self.a*(self.a>0)/self.nbDraws))
        
    def secondChoice(self):
        return np.argsort(self.cumReward/self.nbDraws+np.sqrt(self.a*(self.a>0)/self.nbDraws))[-2]
        
