import numpy as np 
from numpy.random import choice
from copy import deepcopy
 
 
class ETC_OCUCB_2():
     
    def __init__(self, nbArms, eta=2, rho=1/2, alpha=np.sqrt(2), C = 10, epoch_ratio = 2):
        self.nbArms    = nbArms
        self.eta       = eta
        self.rho       = rho
        self.alpha     = alpha
        self.epoch_p_ratio = epoch_ratio 
        self.C = C
 
    def startGame(self):
        self.t         = 0
        self.nbEpoch_pull  = 0
        self.epoch_elim = self.nbArms*self.C
        self.epoch_pull = 1
        self.PullReward = np.zeros(self.nbArms)
        self.ElimReward = np.zeros(self.nbArms)
        self.FreeObs    = np.ones(self.nbArms)
        self.nbPull     = np.zeros(self.nbArms)
        self.arms_pull      = list(range(self.nbArms))
        self.arms_elim      = list(range(self.nbArms))
        self.init  = True
    
    def update_epoch(self,nbEpoch):
        return np.ceil(self.epoch_p_ratio**(self.epoch_p_ratio**nbEpoch))

    def armElimination(self):
        mean   = self.ElimReward/self.FreeObs
         
        Index1 = mean - self.alpha*np.sqrt(np.log( ((self.update_epoch(self.nbEpoch_pull+1)**(3/2))*np.log(self.update_epoch(self.nbEpoch_pull+1)))/self.FreeObs)/self.FreeObs)
         
        Y      = max(Index1[self.arms_elim])
         
        for i in self.arms_elim :
            if mean[i] + self.alpha*np.sqrt(np.log( ((self.update_epoch(self.nbEpoch_pull+1)**(3/2))*np.log(self.update_epoch(self.nbEpoch_pull+1)))/self.FreeObs)/self.FreeObs)[i] < Y :
                self.arms_elim.remove(i)
                
    def armUpdate(self):
        self.arms_pull = deepcopy(self.arms_elim)
        #if self.arms_pull != list(range(self.nbArms)) :
            #print(self.arms_pull,'m=',self.nbEpoch_pull,'horizon =',self.epoch_pull)
 
    def getRewardPull(self, arm, reward):
        self.PullReward[arm] += reward
        self.nbPull[arm] += 1
         
    def getRewardObs(self, arm, reward):
        self.ElimReward[arm] += reward
        self.FreeObs[arm] += 1
     
    def choice_argmax(self, index, arms):
        Index = index
        a = min(Index)
         
        for j in range(self.nbArms):
            if not (j in arms):
                Index[j] = a-1
                 
        b = np.argsort(Index)[::-1]
        c = []
         
        for j in b[0::] :
            if j in arms:
                c.append(j)
                break
                 
        if c[0] < self.nbArms:
            for j in b[c[0]+1::]:
                if Index[j] >= Index[b[0]]:
                    c.append(j)
         
        return choice(np.array(c))
     
    def exploration_term(self):
        B = 0
         
        for j in range(self.nbArms):
            B += np.minimum(self.nbPull, (self.nbPull[j]**self.rho)*self.nbPull**(1-self.rho))
             
        B = np.maximum(np.exp(1), np.log(self.t+self.nbArms), (self.t+self.nbArms)*np.log(self.t+self.nbArms)/B)
         
        return np.sqrt(2*self.eta*np.log(B)/self.nbPull)
     
    def choice(self):
         
        if self.init == True :
            It, Jt = self.t, self.nbArms + 1
             
        else:
            index = (self.PullReward/self.nbPull) + self.exploration_term()
             
            It = self.choice_argmax(index, self.arms_pull)
            Jt = choice(self.arms_elim)
            
            if self.t == self.epoch_elim:
                card_elim = len(self.arms_elim)
                self.armElimination()
                if len(self.arms_elim) < card_elim :
                    self.epoch_elim = self.epoch_elim + len(self.arms_elim)*self.C
                else :
                    self.epoch_elim = self.epoch_elim + card_elim*self.C
                 
            if self.t >= self.epoch_pull:
                self.nbEpoch_pull =self.nbEpoch_pull+1
                self.epoch_pull = self.update_epoch(self.nbEpoch_pull)
                self.armUpdate()
                self.arms_elim = list(range(self.nbArms)) 
                 
        if self.t == self.nbArms-1 and self.init == True :
            self.init = False
            self.t = 0
             
        else:
            self.t = self.t + 1
             
        return It, Jt