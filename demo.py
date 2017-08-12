# -*- coding: utf-8 -*-

__author__ = "Garcelon, Evrard"

from environnement.MAB import MAB
from distribution.Bernoulli import Bernoulli
import numpy as np
import pylab as plt

from policy.UCB1_Uniforme import UCB1_Uniforme
from policy.UCB_SE import UCB_SE
from policy.MOSS1 import MOSS1
from policy.MOSS_SE import MOSS_SE
from policy.UCB_SE_Pot import *
#from policy.UCB1_Potentiel_SE import UCB1_Potentiel_SE
from policy.UCB1_Potentiel import UCB1_Potentiel
from environnement.Evaluation import *



colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'orange', 'gray', 'yellow']
graphic = 'yes'
scenario = 0
q=10
nbRep = 100
horizon = 10**4
epsilon = 1/np.log(horizon)

if scenario == 0: 
    expectation = np.array([0.9, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1])
    a = max(expectation)
    delta = a-expectation
    env = MAB(np.array([Bernoulli(p) for p in expectation]),delta)
elif scenario == 1:
    expectation = np.array([0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
    a = max(expectation)
    delta = a-expectation
    env = MAB(np.array([Bernoulli(p) for p in expectation]),delta)
elif scenario == 2:
    expectation = np.array([0.1, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02])
    a = max(expectation)
    delta = a-expectation
    env = MAB(np.array([Bernoulli(p) for p in expectation]),delta)
else :
    expectation = np.array([0.55, 0.45, 0.35, 0.25, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05])
    a = max(expectation)
    delta = a-expectation
    env = MAB(np.array([Bernoulli(p) for p in expectation]),delta)

policies = [UCB1_Uniforme(env.nbArms),UCB_SE_Pot(env.nbArms,horizon),UCB_SE(env.nbArms, horizon),MOSS1(env.nbArms,horizon),UCB1_Potentiel(env.nbArms)] 
tsav = np.linspace(0,horizon-1,horizon,dtype = int)

if graphic == 'yes':
    plt.figure(scenario)

k=0
for policy in policies:
    ev = Evaluation(env, policy, nbRep, horizon, epsilon, delta, tsav)
    print(ev.meanNbDraws())
    meanRegret = ev.meanRegret()
    qRegret = ev.quantile(q)
    QRegret = ev.quantile(100-q)
    if graphic == 'yes':
        plt.semilogx(tsav, meanRegret, color = colors[k])
        plt.fill_between(tsav, qRegret, QRegret, alpha=0.15, linewidth=1.5, color=colors[k])
        plt.xlabel('Time')
        plt.ylabel('Regret')
    k = k+1

if graphic == 'yes':
    plt.legend([policy.__class__.__name__ for policy in policies], loc=0)
    plt.title('Average regret for various policies with epsilon = 1/ln(T) '.format(epsilon))
    plt.show()
