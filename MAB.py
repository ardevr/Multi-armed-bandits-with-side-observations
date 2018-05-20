import numpy.random as npr
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
            Z = int((npr.rand(1) <= epsilon) * 1)
            It,Jt = policy.choice()
            reward1 = self.arms[It].draw()
            result.store(t, It)
            policy.getRewardPull(It, reward1)
            if Z==1 and t>=self.nbArms and Jt < self.nbArms +1:
                if Jt != It :
                    reward2 = self.arms[Jt].draw()
                else :
                    reward2 = reward1
                policy.getRewardObs(Jt, reward2)
                result.storeObs(t,Jt)
        return result
                
            