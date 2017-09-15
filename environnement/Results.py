# -*- coding: utf-8 -*-
'''Utility class for handling the results of a Multi-armed Bandits experiment with side observations. Inspired by the work of O.Cappé and A.Garivier'''

__author__ = "Garcelon, Evrard"

import numpy as np

class Result:
    """The Result class for analyzing the output of bandit experiments."""
    def __init__(self, nbArms,Delta,T):
        self.nbArms = nbArms
        self.choices = np.zeros(T, dtype = int)
        self.delta = Delta 
        self.regret = np.zeros(T)
    
    def store(self, t, choice):
        self.choices[t] = choice
        if t == 0 :
            self.regret[t] = self.delta[choice]
        else :
            self.regret[t] = self.regret[t-1] + self.delta[choice]
    
    def getNbPulls(self):
        nbPulls = np.zeros(self.nbArms)
        for choice in self.choices:
            nbPulls[choice] += 1
        return nbPulls
    
    def getRegret(self):
        return self.regret
    