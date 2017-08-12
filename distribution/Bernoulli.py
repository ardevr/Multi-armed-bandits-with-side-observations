# -*- coding: utf-8 -*-
'''Bernoulli distributed arm.'''

import numpy as np
class Bernoulli():

    def __init__(self, p): 
        self.p = p
        self.expectation = p
        
    def draw(self):
        return (np.random.rand(1) <= (self.p)) * 1.
    def expectation(self):
        return self.p