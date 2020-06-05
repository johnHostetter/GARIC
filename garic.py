# -*- coding: utf-8 -*-
"""
Created on Fri May 29 23:01:21 2020

@author: jhost
"""

import copy
import math
import random
import numpy as np

class GARIC():
    """ Generalized Approximate Reasoning Intelligent Controller """
    def __init__(self, inputVariables, outputVariables, rules, h):
        self.aen = AEN(len(inputVariables), h)
        self.asn = ASN(inputVariables, outputVariables, rules)
        self.sam = SAM()
    def play(self, x):
        self.aen.X.append(x) # state at time step t + 1
        
class AEN():
    """ Action Evaluation Network """
    def __init__(self, n, h):
        self.n = n # the number of inputs
        self.h = h # the number of neurons in the hidden layer
        self.beta = 0.5 # constant that is greater than zero
        self.gamma = 0.9 # discount rate between 0 and 1
        self.X = [] # the history of visited states
        self.A = [] # the history of previous weights, a
        self.B = [] # the history of previous weights, b
        self.C = [] # the history of previous weights, c
        self.R = [] # the history of actual rewards, r
        self.R_hat = [] # the history of internal reinforcements, r^
        self.__weights(n, h)
    def __weights(self, n, h):
        """ Initializes all the weights in the action evaluation network 
        with random values. """
        a = np.array([0.0]*((n + 1)*h)).reshape(n + 1, h) # plus one for the bias input weight (input to hidden layer weights)
        b = np.array([0.0]*(n + 1)) # plus one for the bias input weight (input to output weights)
        c = np.array([0.0]*(h)) # weights for hidden layer, Y
        # random initialization of the input to hidden layer weights
        for row in range(self.n + 1):
            for col in range(self.h):
                a[row, col] = random.random()
        # random initialization of the input to output weights
        for idx in range(n + 1):
            b[idx] = random.random()
        # random initialization of the hidden layer to output weights
        for idx in range(h):
            c[idx] = random.random()
        # append weights to the history of the action evaluation network
        self.A.append(a)
        self.B.append(b)
        self.C.append(c)
    def __y(self, i, t, t_next):
        """ Calculates the weighted sum of the ith hidden layer neuron
        with weights at time step t and input at time step t + 1. """ 
        s = 0.0
        for j in range(self.n):
            inpt = self.X[t_next][j] # the input at time t + 1
            weight = self.A[t][j, i] # the weight at time t
            s += weight * inpt
        return self.__g(s)
    def __g(self, s):
        """ Sigmoid activation function. """ 
        return 1 / (1 + math.pow(np.e, -s))
    def v(self, t, t_next):
        """ Determines the value of the provided current state. """
        inpts_sum = 0.0
        hidden_sum = 0.0
        for i in range(n):
            inpts_sum += self.B[t][i] * self.X[t_next][i] # the weight, b, at time t and the input, x, at time t + 1
        for i in range(n):
            hidden_sum += self.C[t][i] * self.__y(i, t, t_next) # the weight, c, at time t and the input, y, at time t + 1
        return inpts_sum + hidden_sum
    def r_hat(self, t):
        """ Calculate internal reinforcement given a state, x, and an
        actual reward, r, received at time step t + 1. """
        if len(self.X) <= 1: # start state
            val = 0
        elif self.X == None: # IMPLEMENT: fail state
            val = self.R[t + 1] - self.v(t, t)
        else:
            val = self.R[t + 1] - self.gamma * self.v(t, t + 1) - self.v(t, t)
        self.R_hat.append(val)
        return val
    def backpropagation(self, t):
        """ Updates the weights of the action evaluation network. """
        # update the weights, b
        b_next_t = np.array([0.0]*len(self.B[t]))
        for i in range(len(self.B[t])):
            b_next_t[i] = self.B[t][i] + self.beta * self.R_hat[t + 1] * self.X[t][i]
        self.B.append(b_next_t)
        # update the weights, c
        c_next_t = np.array([0.0]*len(self.C[t]))
        for i in range(len(self.C[t])):
            c_next_t[i] = self.C[t][i] + self.beta * self.R_hat[t + 1] * self.__y(i, t, t)
        self.C.append(c_next_t)
        # update the weights, a
        a_next_t = copy.deepcopy(self.A[t])
        for j in range(len(self.A[t])):
            for i in range(len(self.A[t][j])):
                a_next_t[j][i] = self.A[t][j][i] + self.beta * self.R_hat[t + 1] * self.__y(i, t, t) * (1 - self.__y(i, t, t))*np.sign(self.C[t][i])*self.X[t][j]
        self.A.append(a_next_t)
    
class ASN():
    """ Action Selection Network """
    def __init__(self, inputVariables, outputVariables, rules):
        self.X = []
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
    def forward(self, t):
        """ Completes a forward pass through the Action Selection Network
        provided a given input state. """
        self.o1 = self.X[t]
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
    def dz_dcV(self, w):
        return 1
    def dz_dsVR(self, w):
        return 0.5 * (1 - w)
    def dz_dsVL(self, w):
        return -1 * (1 - w)
    def dF_dpV(self):
        p = []
        cV = 0.0
        sVR = 0.0
        sVL = 0.0
        # forward pass from layer 2 to layer 3
        o3activation = np.array([0.0]*len(self.rules))
        for i in range(len(self.rules)):
            rule = self.rules[i]
            o3activation[i] = rule.degreeOfApplicability(self.k, self.X[len(self.X) - 1])
        for j in range(len(self.consequents)):
            rulesIndexes = np.where(self.o3o4Weights[:,j] == 1.0)[0]
            for ruleIndex in rulesIndexes: 
                cV += o3activation[ruleIndex] * self.dz_dcV(o3activation[ruleIndex])
                sVR += o3activation[ruleIndex] * self.dz_dsVR(o3activation[ruleIndex])
                sVL += o3activation[ruleIndex] * self.dz_dsVL(o3activation[ruleIndex])
            cV /= sum(o3activation)
            sVR /= sum(o3activation)
            sVL /= sum(o3activation)
            p.extend([cV, sVR, sVL])
        return np.array(p)
    def dz_dwr(self):
        pass
    def dF_dwr(self):
        pass
    def dwr_dmuj(self):
        pass
    def dF_dmuv(self):
        pass
    def dv_dpV(self):
        return self.dv_dF() * self.dF_dpV()
    def delta_p(self, t, sam, aen):
        return self.eta * sam.s(t) * aen.R_hat[t] * self.dv_dpV()
    def backpropagation(self, aen, t):
        dv_dF_numerator = aen.v(t, t) - aen.v(t - 1, t - 1)
        dv_dF_denominator = self.forward(t) - self.forward(t - 1)
        dv_dF = dv_dF_numerator / dv_dF_denominator
        return dv_dF

class SAM():
    """ Stochastic Action Modifier """
    def __init__(self):
        self.Fs = []
        self.F_primes = []
        self.R_hat = []
    def sigma(self, r_hat):
        """ Given the internal reinforcement from time step t - 1 """
        return math.exp(r_hat)
    def F_prime(self, t):
        """ The actual recommended action to be applied to the system. """
        mu = self.Fs[t]
        sigma = self.sigma(self.R_hat[t - 1])
        samples = 1
        return np.random.normal(mu, sigma, samples)[0]
    def s(self, t):
        """ Calculates the perturbation at each time step and is simply
        the normalized deviation from the action selection network's 
        recommended action. """
        return (self.F_prime(t) - self.Fs[t]) / self.sigma(self.R_hat[t - 1])
    
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
        """ The parameter idx is the index of the corresponding input/output, 
        name is the linguistic variable's name, and terms are the values that 
        variable can take on. """
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
aen.X.append(x)
aen.X.append(x)
aen.R.append(0)
aen.R.append(1)
print(aen.v(0, 1))
print(aen.r_hat(0))
print(aen.r_hat(0))
print(aen.backpropagation(0))

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

asn.X.append(state)
print(asn.forward(0))
print(asn.dF_dpV())
print(len(asn.dF_dpV()))

sam = SAM()

eta = 0.3
t = 0
sam.R_hat = aen.R_hat
sam.Fs = [asn.forward(0), asn.forward(0)]
pV = eta * sam.s(t) * aen.R_hat[t] * np.sign(asn.backpropagation(aen, t))