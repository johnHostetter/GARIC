# -*- coding: utf-8 -*-
"""
Created on Fri May 29 23:01:21 2020

@author: jhost
"""

import copy
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
    
class ASN():
    """ Action Selection Network """
    def __init__(self, inputVariables, outputVariables, rules):
        self.k = 3.14 # the degree of hardness for softmin
        self.inputVariables = inputVariables
        self.outputVariables = outputVariables
        self.o1 = np.array([0.0]*len(inputVariables)) # input layer
        self.antecedents = self.__antecedents() # generates the antecedents layer
        self.o1o2Weights = self.__o2() # assigns the weights between input layer and terms layer
        self.rules = rules # generates the rules layer
        self.o2o3Weights = self.__o3() # assigns the weights between antecedents layer and rules layer
        self.consequents = self.__consequents() # generates the consequents layer
        self.o3o4Weights = self.__o4() # assigns the weights between rules layer and consequents layer
    def __antecedents(self):
        """ Generates the list of terms to be used in the second layer. """
        terms = []
        for variable in self.inputVariables:
            terms.extend(variable.terms)
        return terms
#    def __rules(self):
#        """ Generates the list of rules to be used in the third layer. """
#        return self.rules
    def __o2(self):
        """ Assigns the weights between input layer and terms layer.
        A weight of '1' is assigned if the connection exists, a weight
        of '0' is assigned if the connection does not exist. """
        weights = np.array([0.0]*(len(self.inputVariables)*len(self.antecedents))).reshape(len(self.inputVariables), len(self.antecedents))
        for row in range(len(self.inputVariables)):
            variable = self.inputVariables[row]
            for term in variable.terms:
                col = self.antecedents.index(term)
                weights[row, col] = 1
        return weights
    def __o3(self):
        """ Assigns the weights between antecedents layer and rules layer.
        A weight of '1' is assigned if the connection exists, a weight of '0' is
        assigned if the connection does not exist. """
        weights = np.array([0.0]*(len(self.antecedents)*len(self.rules))).reshape(len(self.antecedents), len(self.rules))
        for row in range(len(self.antecedents)):
            term = self.antecedents[row]
            for col in range(len(self.rules)):
                rule = self.rules[col]
                if term in rule.antecedents:
                    weights[row, col] = 1
        return weights
    def __consequents(self):
        terms = []
        for variable in self.outputVariables:
            terms.extend(variable.terms)
        return terms
    def __o4(self):
        """ Assigns the weights between rules layer and consequents layer.
        A weight of '1' is assigned if the connection exists, a weight of '0' is
        assigned if the connection does not exist. """
        weights = np.array([0.0]*(len(self.rules)*len(self.consequents))).reshape(len(self.rules), len(self.consequents))
        for row in range(len(self.rules)):
            rule = self.rules[row]
            for col in range(len(self.consequents)):
                consequent = self.consequents[col]
                if consequent in rule.consequents:
                    weights[row, col] = 1
        return weights
    def forward(self, state):
        """ Completes a forward pass through the Action Selection Network
        provided a given input state. """
        self.o1 = state
        o2activation = copy.deepcopy(self.o1o2Weights)
        # forward pass from layer 1 to layer 2
        for i in range(len(self.o1)):
            o2 = np.where(self.o1o2Weights[i]==1.0)[0] # get all indexes that this input maps to
            for j in o2:
                antecedent = self.antecedents[j]
                deg = antecedent.degree(self.o1[i])
                o2activation[i, j] = deg
        # forward pass from layer 2 to layer 3
        o3activation = np.array([0.0]*len(self.rules))
        for i in range(len(self.rules)):
            rule = self.rules[i]
            o3activation[i] = rule.degreeOfApplicability(self.k, state)
        # forward pass from layer 3 to layer 4
        o4activation = np.array([0.0]*len(self.consequents))
        for col in range(len(self.consequents)):
            consequent = self.consequents[col]
            rulesIndexes = np.where(self.o3o4Weights[col]==1.0)[0] # get all rules that map to this consequent
            rulesSum = 0.0
            rulesSquaredSum = 0.0
            # get sum
            for row in rulesIndexes:
                rulesSum += o3activation[row]
            # get squared sum
            for row in rulesIndexes:
                rulesSquaredSum += math.pow(o3activation[row], 2)
            o4activation[col] = (consequent.center + 0.5 * (consequent.rightSpread - consequent.leftSpread)) * rulesSum - 0.5 * (consequent.rightSpread - consequent.leftSpread) * rulesSquaredSum
        # forward pass from layer 4 to layer 5
        F = sum(o4activation) / sum(o3activation)
        return F
        
class Term():
    """ a linguistic term for a linguistic variable """
    def __init__(self, label, center, leftSpread, rightSpread):
        self.label = label
        self.center = center
        self.leftSpread = leftSpread
        self.rightSpread = rightSpread
    def degree(self, x):
        """ degree of membership using triangular membership function """
        if self.center <= x and x <= (self.center + self.rightSpread):
            return 1 - abs(x - self.center) / self.rightSpread
        elif (self.center - self.leftSpread) <= x and x < self.center:
            return 1 - abs(x - self.center) / self.leftSpread
        else:
            return 0

class Variable():
    def __init__(self, idx, name, terms):
        """ idx is the index of the corresponding input/output, name is the
        linguistic variable's name, and terms are the values that 
        variable can take on """
        self.idx = idx
        self.name = name
        self.terms = terms
        
class Rule():
    def __init__(self, antecedents, consequents):
        """ Antecedents is an ordered list of terms in respect to the order of 
        input indexes and consequents is an ordered list of terms in respect 
        to the order of output indexes. """
        self.antecedents = antecedents
        self.consequents = consequents
    def degreeOfApplicability(self, k, inpts):
        """ Determines the degree of applicability for this rule. """
        numerator = 0.0
        denominator = 0.0
        for idx in range(len(inpts)):
            antecedent = self.antecedents[idx]
            if antecedent != None:
                mu = antecedent.degree(inpts[idx])
                numerator += mu * math.pow(math.e, (-k * mu))
                denominator += math.pow(math.e, (-k * mu))
        return numerator / denominator

n = 4
h = 10

# build AEN

x = np.array([0.0]*(n + 1)) # plus one for the bias input
    
# simulate state from environment
for idx in range(n + 1):
    x[idx] = random.random()
    
aen = AEN(n, h)
print(aen.v(x))

# build ASN

# antecedent terms
po1 = Term('PO1', 0.3, 0.3, -1)
ze1 = Term('ZE1', 0.0, 0.3, 0.3)
ne1 = Term('NE1', -0.3, -1, 0.3)
vs1 = Term('VS1', 0.0, 0.05, 0.05)
po2 = Term('PO2', 1.0, 1.0, -1.0)
ze2 = Term('ZE2', 0, 1.0, 1.0)
ne2 = Term('NE2', -1.0, -1.0, 1.0)
vs2 = Term('VS2', 0.0, 0.1, 0.1)
po3 = Term('PO3', 0.5, 0.5, -1.0)
ne3 = Term('NE3', -0.5, -1.0, 0.5)
po4 = Term('PO4', 1.0, 1.0, -1.0)
ne4 = Term('NE4', -1.0, -1.0, 1.0)
ps4 = Term('PS4', 0.0, 0.01, 1.0)
ns4 = Term('NS4', 0.0, 1.0, 0.01)

# consequent terms
pl = Term('PL', 20.0, 5.0, -1.0)
pm = Term('PM', 10.0, 5.0, 6.0)
ps = Term('PS', 5.0, 4.0, 5.0)
pvs = Term('PVS', 1.0, 1.0, 1.0)
nl = Term('NL', -20.0, -1.0, 5.0)
nm = Term('NM', -10.0, 6.0, 5.0)
ns = Term('NS', -5.0, 5.0, 4.0)
nvs = Term('NVS', -1.0, 1.0, 1.0)
ze = Term('ZE', 0.0, 1.0, 1.0)

# antecedent variables
var0 = Variable(0, 'Cart Position', [po3, ne3])
var1 = Variable(1, 'Cart Velocity', [po4, ps4, ne4, ns4])
var2 = Variable(2, 'Pole Angle', [po1, ze1, ne1, vs1])
var3 = Variable(3, 'Pole Velocity At Tip', [po2, ze2, ne2, vs2])

# consequent variables
var4 = Variable(0, 'Force', [pl, pm, ps, pvs, nl, nm, ns, nvs, ze])

inputVariables = [var0, var1, var2, var3]
outputVariables = [var4]

rules = [
        Rule([po1, po2, None, None], [pl]),
        Rule([po1, ze2, None, None], [pm]),
        Rule([po1, ne2, None, None], [ze]),
        Rule([ze1, po2, None, None], [ps]),
        Rule([ze1, ze2, None, None], [ze]),
        Rule([ze1, ne2, None, None], [ns]),
        Rule([ne1, po2, None, None], [ze]),
        Rule([ne1, ze2, None, None], [nm]),
        Rule([ne1, ne2, None, None], [nl]),
        Rule([vs1, vs2, po3, po4], [ps]),
        Rule([vs1, vs2, po3, ps4], [pvs]),
        Rule([vs1, vs2, ne3, ne4], [ns]),
        Rule([vs1, vs2, ne3, ns4], [nvs])
        ]

asn = ASN(inputVariables, outputVariables, rules)
state = None
while True:
    state = np.array([0.0]*(4)) # plus one for the bias input
    
    # simulate state from environment
    for idx in range(n):
        state[idx] = random.random()
    print(rules[0].degreeOfApplicability(2, state))
    break

print(asn.forward(state))