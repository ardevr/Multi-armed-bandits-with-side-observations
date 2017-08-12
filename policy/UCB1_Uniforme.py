# -*- coding: utf-8 -*-
'''The UCB1-Uniforme policy.
Reference = Bandits with Side Observations [GARCELON Evrard]'''

__author__ = "Garcleon, Evrard"

import numpy as np 
import random

class UCB1_Uniforme():
    
    def __init__(self, nbArms):
        self.nbArms  = nbArms
        self.cumReward= np.zeros(self.nbArms)
        self.nbDraws = np.zeros(self.nbArms)
        self.t = 0
        
    def startGame(self):
        self.t       = 0
        self.cumReward= np.zeros(self.nbArms)
        self.nbDraws = np.zeros(self.nbArms)

    def getReward(self,t, arm, reward):
        self.cumReward[arm] += reward
        self.nbDraws[arm]+=1
        self.t = t
        
    def getRewardDouble(self, t, arm1, arm2, reward1, reward2):
        self.cumReward[arm1]   += reward1
        self.cumReward[arm2]   += reward2
        self.nbDraws[arm1] += 1
        self.nbDraws[arm2] += 1
        self.t = t
        
    def choice(self):
        if self.t < self.nbArms:
            return self.t
        else:
            return np.argmax((self.cumReward/self.nbDraws)+np.sqrt(2*np.log(self.t+1)/self.nbDraws))
        
    def secondChoice(self):
        return random.randint(0,self.nbArms-1)
        
