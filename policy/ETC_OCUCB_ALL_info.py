import numpy as np 
from numpy.random import choice
from copy import deepcopy
 
 
class ETC_OCUCB_2_all_info():
     
    def __init__(self, nbArms, eta=2, rho=1/2, alpha=1, C = 10):
        self.nbArms    = nbArms
        self.eta       = eta
        self.rho       = rho
        self.alpha     = alpha
        self.epoch_p_ratio = 2
        self.C = C
 
    def startGame(self):
        self.t         = 0
        self.nbEpoch_pull  = 0
        self.epoch_elim = self.nbArms*self.C
        self.epoch_pull = 1
        self.cumReward  = np.zeros(self.nbArms)
        self.nbObs      = np.zeros(self.nbArms)
        self.arms_pull      = list(range(self.nbArms))
        self.arms_elim      = list(range(self.nbArms))
        self.init  = True
    
    def update_epoch(self,nbEpoch):
        return np.ceil(self.epoch_p_ratio**(self.epoch_p_ratio**nbEpoch))

    def armElimination(self):
        mean   = self.cumReward/self.nbObs
         
        Index1 = mean - self.alpha*np.sqrt(np.log( ((self.update_epoch(self.nbEpoch_pull+1)**(3/2))*np.log(self.update_epoch(self.nbEpoch_pull+1)))/self.nbObs)/self.nbObs)
        Y      = max(Index1[self.arms_elim])
         
        for i in self.arms_elim :
            if mean[i] + self.alpha*np.sqrt(np.log( ((self.update_epoch(self.nbEpoch_pull+1)**(3/2))*np.log(self.update_epoch(self.nbEpoch_pull+1)))/self.nbObs)/self.nbObs)[i] < Y :
                self.arms_elim.remove(i)
                
    def armUpdate(self):
        self.arms_pull = deepcopy(self.arms_elim)
        #if self.arms_pull != list(range(self.nbArms)) :
            #print(self.arms_pull,'m=',self.nbEpoch_pull,'horizon =',self.epoch_pull)
 
    def getRewardPull(self, arm, reward):
        self.cumReward[arm] += reward
        self.nbObs[arm] += 1
         
    def getRewardObs(self, arm, reward):
        self.cumReward[arm] += reward
        self.nbObs[arm] += 1
     
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
            B += np.minimum(self.nbObs, (self.nbObs[j]**self.rho)*self.nbObs**(1-self.rho))
             
        B = np.maximum(np.exp(1), np.log(self.t+self.nbArms), (self.t+self.nbArms)*np.log(self.t+self.nbArms)/B)
         
        return np.sqrt(2*self.eta*np.log(B)/self.nbObs)
     
    def choice(self):
         
        if self.init == True :
            It, Jt = self.t, self.nbArms + 1
             
        else:
            index = (self.cumReward/self.nbObs) + self.exploration_term()
             
            It = self.choice_argmax(index, self.arms_pull)
            arms = deepcopy(self.arms_elim)
            try :
                arms.remove(It)
            except ValueError :
                None
            if len(arms) > 0 :
                Jt = choice(arms)
            else :
                Jt = self.nbArms+1
            
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