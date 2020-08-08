# -*- coding: utf-8 -*-
"""
Created on Fri May 29 23:01:21 2020

@author: jhost
"""

import copy
import math
import random
import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(0)

class GARIC():
    """ Generalized Approximate Reasoning Intelligent Controller """
    def __init__(self, inputVariables, outputVariables, rules, h, lower=-25, upper=25):
        self.X = [] # the history of visited states
        self.R = [] # the history of actual rewards, r
        self.R_hat = [] # the history of internal reinforcements, r^
        self.Fs = [] # the history of recommended actions, F
        self.F_primes = [] # the history of actual actions, F'
        self.aen = AEN(self.X, self.R, self.R_hat, len(inputVariables), h)
        self.asn = ASN(self.X, self.Fs, self.R_hat, inputVariables, outputVariables, rules)
        self.sam = SAM(self.Fs, self.F_primes, self.R_hat, lower, upper)
    def action(self, t):
        F = self.asn.forward(t)
        self.Fs.append(F)
#        print('recommended F: %s' % F)
        F_prime = self.sam.F_prime(t)
        self.F_primes.append(F_prime)
        if np.isnan(F_prime):
            print('here')
        return F_prime
        
class AEN():
    """ Action Evaluation Network """
    def __init__(self, X, R, R_hat, n, h):
        self.X = X
        self.R = R
        self.R_hat = R_hat
        self.bias = 1 # random bias
        self.n = n # the number of inputs
        self.h = h # the number of neurons in the hidden layer
        self.beta = 0.3 # learning rate constant that is greater than zero
        self.gamma = 0.9 # discount rate between 0 and 1
        self.A = [] # the history of previous weights, a
        self.B = [] # the history of previous weights, b
        self.C = [] # the history of previous weights, c
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
                a[row, col] = random.uniform(-0.1, 0.1)
        # random initialization of the input to output weights
        for idx in range(n + 1):
            b[idx] = random.uniform(-0.1, 0.1)
        # random initialization of the hidden layer to output weights
        for idx in range(h):
            c[idx] = random.uniform(-0.1, 0.1)
        # append weights to the history of the action evaluation network
        self.A.append(a)
        self.B.append(b)
        self.C.append(c)
    def y(self, i, t, t_next):
        """ Calculates the weighted sum of the ith hidden layer neuron
        with weights at time step t and input at time step t + 1. """ 
        s = 0.0
        for j in range(self.n):
            inpt = self.X[t_next][j] # the input at time t + 1
            weight = self.A[t][j, i] # the weight at time t
            s += weight * inpt
        return self.g(s)
    def g(self, s):
        """ Sigmoid activation function. """ 
        return 1 / (1 + math.pow(np.e, -s))
    def v(self, t, t_next):
        """ Determines the value of the provided current state. """
        inpts_sum = 0.0
        hidden_sum = 0.0
        for i in range(self.n):
            inpts_sum += self.B[t][i] * self.X[t_next][i] # the weight, b, at time t and the input, x, at time t + 1
        for i in range(self.n):
            hidden_sum += self.C[t][i] * self.y(i, t, t_next) # the weight, c, at time t and the input, y, at time t + 1
        return inpts_sum + hidden_sum
    def r_hat(self, t, done=False, reset=False):
        """ Calculate internal reinforcement given a state, x, and an
        actual reward, r, received at time step t + 1. """
        if reset == True: # start state
            val = 0
        elif done == True: # fail state
            val = self.R[t + 1] - self.v(t, t)
#            print('%s , %s , %s' % (val, self.R[t+1], self.v(t,t)))
        else: # otherwise
            val = self.R[t + 1] + self.gamma * self.v(t, t + 1) - self.v(t, t)
#            print('%s , %s , %s, %s' % (val, self.R[t+1], self.v(t,t+1), self.v(t,t)))
        self.R_hat.append(val)
        return val
    def backpropagation(self, t):
        """ Updates the weights of the action evaluation network. """
        x = [self.bias]
        x.extend(self.X[t])
        # update the weights, b
        b_next_t = np.array([0.0]*len(self.B[t]))
        for i in range(len(self.B[t])):
            b_next_t[i] = self.B[t][i] + self.beta * self.R_hat[t + 1] * x[i]
        self.B.append(b_next_t)
        # update the weights, c
        c_next_t = np.array([0.0]*len(self.C[t]))
        for i in range(len(self.C[t])):
            c_next_t[i] = self.C[t][i] + self.beta * self.R_hat[t + 1] * self.y(i, t, t)
        self.C.append(c_next_t)
        # update the weights, a
        a_next_t = copy.deepcopy(self.A[t])
        for j in range(len(self.A[t])):
            for i in range(len(self.A[t][j])):
                a_next_t[j][i] = self.A[t][j][i] + self.beta * self.R_hat[t + 1] * self.y(i, t, t) * (1 - self.y(i, t, t))*np.sign(self.C[t][i])*x[j]
        self.A.append(a_next_t)
    
class ASN():
    """ Action Selection Network """
    def __init__(self, X, Fs, R_hat, inputVariables, outputVariables, rules):
        self.X = X
        self.Fs = Fs
        self.R_hat = R_hat
        self.k = 10 # the degree of hardness for softmin
        self.eta = 0.03 # the learning rate
        self.inputVariables = inputVariables
        self.outputVariables = outputVariables
#        self.o1 = np.array([0.0]*len(inputVariables)) # input layer
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
#        print('forward pass %s' % t)
        o2activation = copy.deepcopy(self.o1o2Weights)
        # forward pass from layer 1 to layer 2
        for i in range(len(self.X[t])):
            o2 = np.where(self.o1o2Weights[i]==1.0)[0] # get all indexes that this input maps to
            for j in o2:
                antecedent = self.antecedents[j]
#                print(antecedent.label)
                deg = antecedent.degree(self.X[t][i])
                o2activation[i, j] = deg
#        print('o2activation: %s' % o2activation)
        # forward pass from layer 2 to layer 3
        o3activation = np.array([0.0]*len(self.rules))
        for i in range(len(self.rules)):
            rule = self.rules[i]
            o3activation[i] = rule.degreeOfApplicability(self.k, self.X[t])
#        print('o3activation: %s' % o3activation)
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
#        if sum(o3activation) != 0:
#        print('o4activation: %s' % o4activation)
#        print('sum of o3activation: %s' % sum(o3activation))
#        print('sum of o4activation: %s' % sum(o4activation))
        F = sum(o4activation) / sum(o3activation)
#            self.Fs.append(F)
        if np.isnan(F):
            print('here')
        return F
#        else:
#            self.Fs.append(0.0)
#            return 0.0
    def dmuV_dcV(self, V, x): # producing lots of zeroes
        if V.center <= x and x <= (V.center + V.rightSpread):
            return (x - V.center) / (V.rightSpread * abs(x - V.center))
        elif (V.center - V.leftSpread) <= x and x < V.center:
            return (x - V.center) / (V.leftSpread * abs(x - V.center))
        else:
            return 0
    def dmuV_dsVR(self, V, x):
        if V.center <= x and x <= (V.center + V.rightSpread):
            return (abs(x - V.center)) / math.pow(V.rightSpread, 2)
        elif (V.center - V.leftSpread) <= x and x < V.center:
            return 0
        else:
            return 0
    def dmuV_dsVL(self, V, x):
        if V.center <= x and x <= (V.center + V.rightSpread):
            return 0
        elif (V.center - V.leftSpread) <= x and x < V.center:
            return (abs(x - V.center)) / math.pow(V.leftSpread, 2)
        else:
            return 0
    def dz_dcV(self, w):
        return 1
    def dz_dsVR(self, w):
        return 0.5 * (1 - w)
    def dz_dsVL(self, w):
        return -1 * (1 - w)
    
    
    
    
    
    
    
    
    
    def dF_dpV(self):
        p = []
        cV = 0.0
        sVL = 0.0
        sVR = 0.0
        # forward pass from layer 2 to layer 3
        o3activation = np.array([0.0]*len(self.rules))
        for i in range(len(self.rules)):
            rule = self.rules[i]
            o3activation[i] = rule.degreeOfApplicability(self.k, self.X[len(self.X) - 1])
#        print('o3 activation: %s' % o3activation)
        for j in range(len(self.consequents)):
            rulesIndexes = np.where(self.o3o4Weights[:,j] == 1.0)[0]
            for ruleIndex in rulesIndexes: 
#                print('dz_dcV: %s' % self.dz_dcV(o3activation[ruleIndex]))
#                print('dz_dsVL: %s' % self.dz_dsVL(o3activation[ruleIndex]))
#                print('dz_dsVR: %s' % self.dz_dsVR(o3activation[ruleIndex]))
                cV += o3activation[ruleIndex] * self.dz_dcV(o3activation[ruleIndex])
                sVL += o3activation[ruleIndex] * self.dz_dsVL(o3activation[ruleIndex])
                sVR += o3activation[ruleIndex] * self.dz_dsVR(o3activation[ruleIndex])
            if sum(o3activation) != 0:
                cV /= sum(o3activation)
                sVL /= sum(o3activation)
                sVR /= sum(o3activation)
                p.extend([cV, sVL, sVR])
                cV = 0.0
                sVL = 0.0
                sVR = 0.0
            else:
                p.extend([0.0, 0.0, 0.0])
                cV = 0.0
                sVL = 0.0
                sVR = 0.0
        return np.array(p)
    def z(self, V, r):
        w_r = self.rules[r].degreeOfApplicability(self.k, self.X[len(self.X) - 1])
        return V.center + 0.5 * (V.rightSpread - V.leftSpread) * (1 - w_r)
    
    
    
    
    
    
    
    
    
    # my own, may be okay, may not be
    def dz_dwr(self, V, r):
        return -0.5 * (V.rightSpread - V.leftSpread)
    
    
    
    
    
    
    
    
    def dF_dwr(self, V, r):
        w_r = self.rules[r].degreeOfApplicability(self.k, self.X[len(self.X) - 1])
        numerator = w_r * self.z(V, r) + self.dz_dwr(V, r) - self.Fs[len(self.Fs) - 1]
        denominator = 0.0
#        idx = self.antecedents.index(V)
#        relevantRulesIndexes = np.where(self.o2o3Weights[idx] == 1.0)[0]
#        for ruleIndex in relevantRulesIndexes:
#            rule = self.rules[ruleIndex]
#            denominator += rule.degreeOfApplicability(self.k, self.X[len(self.X) - 1])
        o3activation = np.array([0.0]*len(self.rules))
        for i in range(len(self.rules)):
            rule = self.rules[i]
            o3activation[i] = rule.degreeOfApplicability(self.k, self.X[len(self.X) - 1])
        denominator = sum(o3activation)
        if denominator == 0.0:
            return 0.0
        else:
            if np.isnan(numerator / denominator):
                print('dF_dwr')
                exit
            return numerator / denominator     
    def dwr_dmuj(self, r, j):
        denominator = 0.0
        # get the index of the input that corresponds to the jth term (should only be 1 input)
        inpt_idx = np.where(self.o1o2Weights[:,j] == 1.0)[0][0]
        # calculate the degree of membership of the jth term with its corresponding input
        inpt = self.X[len(self.X) - 1][inpt_idx]
        mu_j = self.antecedents[j].degree(inpt)
        # calculate the degree of applicability for the rth rule given all of the most recent inputs
        w_r = self.rules[r].degreeOfApplicability(self.k, self.X[len(self.X) - 1])
        # calculate the entire product that is the numerator of the quotient
        numerator = math.pow(math.e, -self.k * mu_j)*(1 + self.k*(w_r - mu_j))
        for i in range(len(self.X[len(self.X) - 1]) - 1):
            inpt = self.X[len(self.X) - 1][i]
            mu_i = self.antecedents[i].degree(inpt)
            denominator += math.pow(math.e, -self.k * mu_i)
        if np.isnan(numerator / denominator):
            print('dwr_dmuj')
            exit
        return numerator / denominator
    def dF_dmuV(self, V): # producing large negative/positive values
        val = 0.0
        idx = self.antecedents.index(V)
        relevantRulesIndexes = np.where(self.o2o3Weights[idx] == 1.0)[0]
        for ruleIndex in relevantRulesIndexes:
#            print('dF_dwr: %s' % self.dF_dwr(V, ruleIndex))
#            print('dwr_dmuj: %s' % self.dwr_dmuj(ruleIndex, idx))
            val += self.dF_dwr(V, ruleIndex) * self.dwr_dmuj(ruleIndex, idx)
        if np.isnan(val):
            print('dF_dmuV')
            exit
        return val
    def dv_dF(self, aen, t):
        dv_dF_numerator = aen.v(t, t) - aen.v(t, t - 1)
        dv_dF_denominator = self.forward(t) - self.forward(t - 1)
        if dv_dF_denominator == 0:
            return 1
        else:
            if np.isnan(np.sign(dv_dF_numerator / dv_dF_denominator)):
                print('dv_dF')
                exit
            return np.sign(dv_dF_numerator / dv_dF_denominator)
    def dv_dpV(self):
        if np.isnan(self.dv_dF() * self.dF_dpV()):
            print('dv_dpV')
            exit
        return self.dv_dF() * self.dF_dpV()
    def delta_p(self, t, sam, aen):
        if np.isnan(self.eta * sam.s(t) * aen.R_hat[t] * self.dv_dpV()):
            print('delta_p')
            exit
        return self.eta * sam.s(t) * aen.R_hat[t] * self.dv_dpV()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def backpropagation(self, aen, sam, t):
#        print('sam.s(t): %s' % sam.s(t))
#        print('r^: %s' % self.R_hat[t])
#        print('dv_dF: %s' % self.dv_dF(aen, t))
#        print('dF_dpV: %s' % self.dF_dpV())
#        print('poss: %s' % (self.R_hat[t] * self.dv_dF(aen, t) * self.dF_dpV()))
#        if self.Fs[t] == 0:
#            exit
        delta_pV = self.eta * sam.s(t) * self.R_hat[t] * self.dv_dF(aen, t) * self.dF_dpV() # temporary, just works for consequents
        new_weights = np.array([0.0, 0.0, 0.0]*len(self.consequents)).reshape(len(self.consequents), 3)
        for i in range(len(self.consequents)):
            self.consequents[i].center += delta_pV[0+(i*3)]
            self.consequents[i].leftSpread += delta_pV[1+(i*3)]
            self.consequents[i].rightSpread += delta_pV[2+(i*3)]
            new_weights[i, 0] = self.consequents[i].center
            new_weights[i, 1] = self.consequents[i].leftSpread
            new_weights[i, 2] = self.consequents[i].rightSpread
        # tune antecedents
        print('dv_dF: %s' % self.dv_dF(aen, t))
        if t < 2000:
            for i in range(len(self.antecedents)):
                antecedent = self.antecedents[i]
                idx = np.where(self.o1o2Weights[:, i] == 1.0)[0][0] # there should only be one corresponding input
                x = self.X[len(self.X) - 1][idx]
    #            print('BEFORE: ' + str([self.antecedents[i].label, self.antecedents[i].center, self.antecedents[i].rightSpread, self.antecedents[i].leftSpread]))
#                print('dF_dmuV: %s' % self.dF_dmuV(antecedent))
#                print('dmuV_dcV: %s' % self.dmuV_dcV(antecedent, x))
#                print('delta: %s' % (self.eta * sam.s(t) * self.R_hat[t] * self.dv_dF(aen, t) * self.dF_dmuV(antecedent) * self.dmuV_dcV(antecedent, x)))
                self.antecedents[i].center += self.eta * sam.s(t) * self.R_hat[t] * self.dv_dF(aen, t) * self.dF_dmuV(antecedent) * self.dmuV_dcV(antecedent, x)
                self.antecedents[i].rightSpread += self.eta * sam.s(t) * self.R_hat[t] * self.dv_dF(aen, t) * self.dF_dmuV(antecedent) * self.dmuV_dsVR(antecedent, x)
                self.antecedents[i].leftSpread += self.eta * sam.s(t) * self.R_hat[t] * self.dv_dF(aen, t) * self.dF_dmuV(antecedent) * self.dmuV_dsVL(antecedent, x)
#                print('AFTER: ' + str([self.antecedents[i].label, self.antecedents[i].center, self.antecedents[i].rightSpread, self.antecedents[i].leftSpread]))
            
class SAM():
    """ Stochastic Action Modifier """
    def __init__(self, Fs, F_primes, R_hat, lower, upper):
        self.Fs = Fs
        self.F_primes = F_primes
        self.R_hat = R_hat
        self.lower = lower
        self.upper = upper
    def sigma(self, r_hat):
        """ Given the internal reinforcement from time step t - 1 """
        return math.exp((-1) * r_hat)
    def F_prime(self, t):
        """ The actual recommended action to be applied to the system. """
        mu = self.Fs[t]
        sigma = self.sigma(self.R_hat[t - 1])
        samples = 1
        F_prime = np.random.normal(mu, sigma, samples)[0]
        if F_prime > self.upper or F_prime < self.lower:
            if self.R_hat[t - 1] > 0:
                F_prime = np.random.normal(mu, 1/sigma, samples)[0]
            else:
#                F_prime = np.random.normal(mu, self.upper, samples)[0] # temporary fix?
#                F_prime = np.random.normal(0, 1, samples)[0]
                F_prime = np.random.uniform(-1,1,size=1)[0]
#                F_prime = mu
#        if F_prime < -25:
#            F_prime = -25
#        elif F_prime > 25:
#            F_prime = 25
#        self.F_primes.append(F_prime)
        return F_prime
    def s(self, t):
        """ Calculates the perturbation at each time step and is simply
        the normalized deviation from the action selection network's 
        recommended action. """
        if self.sigma(self.R_hat[t - 1]) == 0:
            return 0.0
        else:
            return (self.F_primes[t] - self.Fs[t]) / self.sigma(self.R_hat[t - 1])
    
class Term():
    """ a linguistic term for a linguistic variable """
    def __init__(self, label, center, leftSpread, rightSpread):
        self.label = label
        self.center = center
        self.leftSpread = leftSpread
        self.rightSpread = rightSpread
    def degree(self, x):
        """ degree of membership using triangular membership function """
#        numerator = (-1) * pow(x - self.center, 2)
#        denominator = 2 * pow(self.leftSpread, 2)
#        return pow(math.e, numerator / denominator)
        if self.center <= x and x <= (self.center + self.rightSpread):
            return 1 - abs(x - self.center) / self.rightSpread
        elif (self.center - self.leftSpread) <= x and x < self.center:
            return 1 - abs(x - self.center) / self.leftSpread
        else:
            return 0
    def __str__(self):
        return'%s: %s, %s, %s' % (self.label, self.center, self.leftSpread, self.rightSpread)

class Variable():
    def __init__(self, idx, name, terms):
        """ The parameter idx is the index of the corresponding input/output, 
        name is the linguistic variable's name, and terms are the values that 
        variable can take on. """
        self.idx = idx
        self.name = name
        self.terms = terms
    def __str__(self):
        output = '%s %s' % (self.idx, self.name)
        for term in self.terms:
            output += str(term) + ' '
        return output
    def graph(self, lower=-20, upper=20):
        for fuzzySet in self.terms:
            x_list = np.linspace(lower, upper, 1000)
            y_list = []
            for x in x_list:
                y_list.append(fuzzySet.degree(x))
            plt.plot(x_list, y_list, color=np.random.rand(3,), label=fuzzySet.label)

        if self.name != None:    
            plt.title('%s Fuzzy Variable' % self.name)
        else:
            plt.title('Unnamed Fuzzy Variable')
        
        plt.axes()
        plt.xlim([lower, upper])
        plt.ylim([0, 1.1])
        plt.xlabel('Elements of Universe')
        plt.ylabel('Degree of Membership')
        plt.legend()
        plt.show()
        
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