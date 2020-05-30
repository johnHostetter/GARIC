# -*- coding: utf-8 -*-
"""
Created on Fri May 29 23:01:21 2020

@author: jhost
"""

import math
import random
import numpy as np

class AEN():
    """ action evaluation network """
    def __init__(self, n, h):
        self.n = n
        self.h = h
        self.x = np.array([0.0]*(n + 1)) # plus one for the bias input
        self.a = np.array([0.0]*((n + 1)*h)).reshape(n + 1, h) # plus one for the bias input weight (input to hidden layer weights)
        self.b = np.array([0.0]*(n + 1)) # plus one for the bias input weight (input to output weights)
        self.c = np.array([0.0]*(h)) # weights for hidden layer, Y
        self.__weights(n, h)
    def __weights(self, n, h):
        """ initializes all the weights in the action evaluation network 
        with random values """
        # random initialization of the input to hidden layer weights
        for row in range(self.n + 1):
            for col in range(self.h):
                self.a[row, col] = random.random()
        # random initialization of the input to output weights
        for idx in range(n + 1):
            self.b[idx] = random.random()
        # random initialization of the hidden layer to output weights
        for idx in range(h):
            self.c[idx] = random.random()
    def __y(self, i):
        """ calculates the weighted sum of the ith hidden layer neuron """ 
        s = 0.0
        for j in range(self.n):
            inpt = self.x[j]
            weight = self.a[i, j]
            s += weight * inpt
        return self.__g(s)
    def __g(self, s):
        """ sigmoid function """ 
        return 1 / (1 + math.pow(np.e, -s))
    def v(self, x):
        """ determines the value of the provided current state """
        self.x = x
        inpts_sum = 0.0
        hidden_sum = 0.0
        for i in range(n):
            inpts_sum += self.b[i] * self.x[i]
        for i in range(n):
            hidden_sum += self.c[i] * self.__y(i)
        return inpts_sum + hidden_sum

n = 4
h = 10

# build ANN

x = np.array([0.0]*(n + 1)) # plus one for the bias input
    
# simulate state from environment
for idx in range(n + 1):
    x[idx] = random.random()
    
aen = AEN(n, h)
print(aen.v(x))