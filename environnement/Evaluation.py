# -*- coding: utf-8 -*-
'''A utility class for evaluating the performance of a policy in multi-armed bandit problems with side observations. Inspired by the work of O.Cappé and A.Garivier'''

__author__ = "Garcelon, Evrard"

import numpy as np

class Evaluation():
  
    def __init__(self, env, pol, nbRepetitions, T, epsilon, Delta, tsav=[]):
        if len(tsav)>0:
            self.tsav = tsav
        else:
            self.tsav = np.arange(T)
        self.env = env
        self.pol = pol
        self.nbRepetitions = nbRepetitions
        self.T = T
        self.nbArms = env.nbArms
        self.epsilon = epsilon
        self.nbPulls = np.zeros((self.nbRepetitions, self.nbArms))
        self.regret = np.zeros((self.nbRepetitions, len(tsav)))
        for k in range(nbRepetitions): 
            if nbRepetitions < 10 or k % (nbRepetitions/10)==0:
                print(k)
            result = env.play(pol, epsilon, T)
            self.nbPulls[k,:] = result.getNbPulls()
            self.regret[k,:] = result.regret[tsav]
            
    def meanNbDraws(self):
        return np.mean(self.nbPulls ,0) 

    def meanRegret(self):
        return np.mean(self.regret,axis=0)
    
    def quantile(self,q):
        return np.percentile(self.regret, q, axis=0)
