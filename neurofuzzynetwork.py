# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 21:38:49 2020

@author: jhost
"""

import math
import copy
import numpy as np
from garic import AEN, SAM
import matplotlib.pyplot as plt

def NFN_gaussianMembership(params, x):
    numerator = (-1) * pow(x - params['center'], 2)
    denominator = 2 * pow(params['sigma'], 2)
    return pow(math.e, numerator / denominator)

class Term():
    """ a linguistic term for a linguistic variable """
    def __init__(self, var, function, params, support=None, label=None):
        """ The 'var' parameter allows this linguistic term to be 
        traced back to its corresponding linguistic variable. 
        The parameter 'function' allows the linguistic term
        to be defined by a variety of applicable functions and is 
        the linguistic term's membership function. The parameter 
        'params' is a dictionary that specifies the parameters of 
        the function argument. The support (optional) specifies
        how many observations were used in creating the membership 
        function. The label (optional) assigns a name to this 
        linguistic term. """
        self.var = var # the ID of the linguistic variable to which this term belongs to
        self.function = function # the membership function
        self.params = params # the corresponding parameters to the membership function
        self.support = support
        self.label = label # the label of the linguistic term
    def degree(self, x):
        """ degree of membership using triangular membership function """
        return self.function(self.params, x)

class Variable():
    def __init__(self, idx, terms, label=None):
        """ The parameter 'idx' is the index of the corresponding input/output, 
        name is the linguistic variable's name, and terms are the values that 
        variable can take on. """
        self.idx = idx # the corresponding input/output or feature of this variable
        self.terms = terms # the linguistic terms for which this linguistic variable is defined over
        self.label = label # the label of the linguistic variable
    def graph(self, lower=-20, upper=20):
        for fuzzySet in self.terms:
            x_list = np.linspace(lower, upper, 1000)
            y_list = []
            for x in x_list:
                y_list.append(fuzzySet.degree(x))
            plt.plot(x_list, y_list, color=np.random.rand(3,), label=fuzzySet.label)

        if self.label != None:    
            plt.title('%s Fuzzy Variable' % self.label)
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
        mus = []
        for idx in range(len(inpts)):
            antecedent = self.antecedents[idx]
            if antecedent != None:
                mu = antecedent.degree(inpts[idx])
                mus.append(mu)
        return min(mus)
#        numerator = 0.0
#        denominator = 0.0
#        for idx in range(len(inpts)):
#            antecedent = self.antecedents[idx]
#            if antecedent != None:
#                mu = antecedent.degree(inpts[idx])
#                numerator += mu * math.pow(math.e, (-k * mu))
#                denominator += math.pow(math.e, (-k * mu))
#        return numerator / denominator
    
class eFL_ACC():
    """ Empirical Fuzzy Logic Actor Critic Controller """
    def __init__(self, inputVariables, outputVariables, rules, h, lower=-25, upper=25):
        self.X = [] # the history of visited states
        self.R = [] # the history of actual rewards, r
        self.R_hat = [] # the history of internal reinforcements, r^
        self.Fs = [] # the history of recommended actions, F
        self.F_primes = [] # the history of actual actions, F'
        self.aen = AEN(self.X, self.R, self.R_hat, len(inputVariables), h)
        self.asn = GenericASN(self.X, self.Fs, self.R_hat, inputVariables, outputVariables, rules)
        self.sam = SAM(self.Fs, self.F_primes, self.R_hat, lower, upper)
    def action(self, t):
        F = self.asn.forward(t)
        self.Fs.append(F)
        F_prime = self.sam.F_prime(t)
        self.F_primes.append(F_prime)
        return F_prime

class GenericASN():
    """ Generic Action Selection Network """
    def __init__(self, X, Fs, R_hat, inputVariables, outputVariables, rules):
        self.X = X
        self.Fs = Fs
        self.R_hat = R_hat
        self.k = 10 # the degree of hardness for softmin
        self.eta = 0.01 # the learning rate
        self.inputVariables = inputVariables
        self.outputVariables = outputVariables
        self.antecedents = self.__antecedents() # generates the antecedents layer
        self.o1o2Weights = self.__o2() # assigns the weights between input layer and terms layer
        self.rules = rules # generates the rules layer
        self.o2o3Weights = self.__o3() # assigns the weights between antecedents layer and rules layer
        self.consequents = self.__consequents() # generates the consequents layer
        self.o3o4Weights = self.__o4() # assigns the weights between rules layer and consequents layer
        self.O4 = {}
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
        o2activation = copy.deepcopy(self.o1o2Weights)
        # layer 2
        for i in range(len(self.X[t])):
            o2 = np.where(self.o1o2Weights[i]==1.0)[0] # get all indexes that this input maps to
            for j in o2:
                antecedent = self.antecedents[j]
                deg = antecedent.degree(self.X[t][i])
                o2activation[i, j] = deg
        # layer 3
        o3activation = np.array([0.0]*len(self.rules))
        for i in range(len(self.rules)):
            rule = self.rules[i]
            o3activation[i] = rule.degreeOfApplicability(self.k, self.X[t])
        # layer 4
        o4activation = np.array([0.0]*len(self.consequents))
        for col in range(len(self.consequents)):
#            consequent = self.consequents[col]
            rulesIndexes = np.where(self.o3o4Weights[col]==1.0)[0] # get all rules that map to this consequent
            f = 0.0
            for row in rulesIndexes:
                f += o3activation[row]
            o4activation[col] = min(1, f)
        self.O4[t] = o4activation
        # layer 5
        f = 0.0
        denominator = 0.0
        for idx in range(len(o4activation)):
            f += (self.consequents[idx].params['center'] * self.consequents[idx].params['sigma'] * o4activation[idx])
            denominator += (self.consequents[idx].params['sigma'] * o4activation[idx])
        a = f / denominator
        return a
        
    def backpropagation(self, aen, sam, t, actual):
        # (1/2) tune consequents
        dv_dF = (aen.v(t, t) - aen.v(t-1, t-1)) / (self.Fs[t] - self.Fs[t-1])
        numerator = 0.0
        denominator = 0.0
        
        o4activation = self.O4[t]
        
#        for consequent in self.consequents:
        for idx in range(len(self.consequents)):
            u_i = o4activation[idx]
            consequent = self.consequents[idx]
            numerator += consequent.params['center'] * consequent.params['sigma'] * u_i
            denominator += consequent.params['sigma'] * u_i
#        print('numerator %s' % numerator)
#        print('denominator %s' % denominator)
#        for consequent in self.consequents:
        for idx in range(len(self.consequents)):
#            print('CENTER BEFORE: %s' % (consequent.params['center']))
            u_i = o4activation[idx]
            consequent = self.consequents[idx]
            consequent.params['center'] += self.eta * np.sign(dv_dF) * ((consequent.params['sigma'] * u_i) / denominator)
#            print('CENTER AFTER: %s' % (consequent.params['center']))
#            print('CENTER BEFORE: %s' % (consequent.params['sigma']))
            consequent.params['sigma'] += self.eta * np.sign(dv_dF) * (((consequent.params['center'] * u_i * denominator) - (numerator * u_i)) / (pow(denominator, 2)))
#            print('CENTER AFTER: %s' % (consequent.params['sigma']))
#        print('DONE')

        # (2/2) tune antecedents
        
        delta_5 = np.sign(dv_dF)
        delta_4 = {} # indexed by consequents
        for idx in range(len(self.consequents)):
            u_i = o4activation[idx]
            consequent = self.consequents[idx]
            delta_4[idx] = delta_5 * (((consequent.params['center'] * consequent.params['sigma'] * denominator) - (numerator * consequent.params['sigma'])) / (pow(denominator, 2)))
        
#        print(delta_4)
        delta_3 = delta_4
#        print(delta_3)
#        o2activation = copy.deepcopy(self.o1o2Weights)
#        # layer 2
#        for i in range(len(self.X[t])):
#            o2 = np.where(self.o1o2Weights[i]==1.0)[0] # get all indexes that this input maps to
#            for j in o2:
#                antecedent = self.antecedents[j]
#                deg = antecedent.degree(self.X[t][i])
#                o2activation[i, j] = deg
                
        for i in range(len(self.X[t])):
            x = self.X[t]
            u_i = x[i]
            o2 = np.where(self.o1o2Weights[i]==1.0)[0] # get all indexes that this input maps to
            for j in o2:
                antecedent = self.antecedents[j]
                a_i = antecedent.degree(u_i) # degree currently includes e^f, so this is actually activation function
#                o2activation[i, j] = deg
                delta_m_ij = a_i * (2 * (u_i - antecedent.params['center'])) / pow(antecedent.params['sigma'], 2)
                delta_sigma_ij = a_i * pow((2 * (u_i - antecedent.params['center'])), 2) / pow(antecedent.params['sigma'], 3)
                
                
                # calculate dE / da_i
                
                # first, find all the rules that this antecedent feeds into
                ruleIndexes = np.where(self.o2o3Weights[j]==1.0)[0] # get all rules that receive input from this antecedent
                # dE / da_i is the summation of q_k
                dE_da_i = 0.0
                # find out if this antecedent is the minimum activation for this rule
                for k in ruleIndexes:
                    q_k = 0.0
                    rule = self.rules[k]
                    deg = rule.degreeOfApplicability(self.k, self.X[t])
                    if deg == a_i:
#                        dE_da_i = 1 # NOT CORRECT, JUST TESTING
                        # find the error of this rule's consequence
                        
                        # find the consequent's index of this rule
                        consequentIndexes = np.where(self.o3o4Weights[k]==1.0)[0]
                        
                        # the error of the kth rule is the summation of the errors of its consequences
                        for consequentIndex in consequentIndexes:
                            q_k += delta_3[consequentIndex]
                    else:
                        q_k = 0.0 # do nothing
                    
                    dE_da_i += q_k
                
                
                
                
                # UPDATE THE WEIGHTS
                antecedent.params['center'] += self.eta * dE_da_i * delta_m_ij
                antecedent.params['sigma'] += self.eta * dE_da_i * delta_sigma_ij