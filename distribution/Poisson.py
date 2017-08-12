# -*- coding: utf-8 -*-
'''Poisson distributed arm '''

__author__ = "Garcelon, Evrard"

from scipy.stats import poisson

class Poisson():
    def __init__(self,p=1):
        self.p = p
    def draw(self):
        return poisson.rvs(self.p,size=1)
    def expectation(self):
        return self.p