# -*- coding: utf-8 -*-
''' Environnement for a multi-armed bandit problem with side observations. Inspired by the work of O.Cappé and A.Garivier'''
__author__ = "Garcelon, Evrard"

import numpy.random
import numpy as np
from environnement.Results import *
class MAB:
    def __init__(self,arms,Delta):
        self.arms = arms
        self.nbArms = len(arms)
        self.delta = Delta
    
    def play(self,policy,epsilon,T):
        policy.startGame()
        result = Result(self.nbArms,self.delta,T)
        for t in range(T):
            Z = int((np.random.rand(1) <= (epsilon)) * 1)
            It = policy.choice() #Choice of the arm played at t
            reward1 = self.arms[It].draw()
            result.store(t, It)
            policy.getReward(t,It, reward1)
            if Z==1 and t>=self.nbArms:
                Jt = policy.secondChoice() #Choice of the second arm played at t if the player has access to a second observation that does not impact the regret 
                if Jt != It:
                    reward2 = self.arms[Jt].draw()
                    policy.getReward(t, Jt, reward2)
        return result
                
            