# -*- coding: utf-8 -*-
'''The UCB1-Uniforme policy'''

__author__ = "Garcleon, Evrard"

import numpy as np 
import random

class UCB1_Uniforme():
    
    def __init__(self, nbArms):
        self.nbArms    = nbArms
        self.cumReward = np.zeros(self.nbArms)
        self.nbDraws   = np.zeros(self.nbArms)
        self.t         = 0
        
    def startGame(self):
        self.t         = 0
        self.cumReward = np.zeros(self.nbArms)
        self.nbDraws   = np.zeros(self.nbArms)

    def getReward(self,t, arm, reward):
        self.cumReward[arm] += reward
        self.nbDraws[arm]   +=1
        self.t               = t+1
                
    def choice(self):
        if self.t < self.nbArms:
            return self.t
        else:
            return np.argmax((self.cumReward/self.nbDraws)+np.sqrt(2*np.log(self.t)/self.nbDraws))
    
    def secondChoice(self):
        return random.randint(0,self.nbArms-1)
        
