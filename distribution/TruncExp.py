# -*- coding: utf-8 -*-
'''Exponentially distributed arm upper bounded by b with b in ]0,1].'''
__author__ = "Garcelon, Evrard"

from scipy.stats import truncexpon

class TruncExp():
    def __init__(self,p=2,b=1):
        self.p = p
        self.b = b
    def draw(self):
        return truncexpon.rvs(self.b,size=1)
    def exptectation(self):
        return 1/self.p
        