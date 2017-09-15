# -*- coding: utf-8 -*-
'''The UCB1-Potentiel policy.'''

__author__ = "Garcleon, Evrard"

import numpy as np 
from numpy.random import choice


class UCB1_Potentiel():
    
    def __init__(self, nbArms, psi = lambda x : np.exp(x), alpha = 1/2):
        self.nbArms     = nbArms
        self.cumReward  = np.zeros(self.nbArms)
        self.nbDraws    = np.zeros(self.nbArms)
        self.psi        = psi
        self.alpha      = alpha
        self.arms       = np.arange(self.nbArms, dtype = int)
        
    def startGame(self):
        self.t          = 0
        self.cumReward  = np.zeros(self.nbArms)
        self.nbDraws    = np.zeros(self.nbArms)

    def getReward(self, t, arm, reward):
        self.cumReward[arm] += reward
        self.nbDraws[arm]   += 1
        self.t               = t+1
    
    def choice(self):
        if self.t<self.nbArms:
            return self.t
        else:
            return np.argmax(self.cumReward/self.nbDraws+np.sqrt(self.alpha*np.log(self.t)/self.nbDraws))
        
    def secondChoice(self):
        self.P = 1/self.psi(self.cumReward/self.nbDraws)
        self.c = np.sum(self.P)
        self.p = self.P/self.c
        return int(choice(self.arms,1,p=self.p))
        
