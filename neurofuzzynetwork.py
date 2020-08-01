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

np.random.seed(0)

def NFN_gaussianMembership(params, x):
    numerator = (-1) * pow(x - params['center'], 2)
    denominator = 2 * pow(params['sigma'], 2)
    return pow(math.e, numerator / denominator)

def NFN_bellShapedMembership(params, x):
    f = -1 * pow((x - params['center']), 2) / pow(params['sigma'], 2)
    return pow(math.e, f)

#def NFN_leftSigmoidMembership(params, x):
#    return 1 / (pow(math.e, np.abs(25/params['sigma']) * (x + params['center'])) + 1)
#
#def NFN_rightSigmoidMembership(params, x):
#    return 1 / (pow(math.e, -np.abs(25/params['sigma']) * (x + params['center'])) + 1)

class Term():
    """ a linguistic term for a linguistic variable """
    def __init__(self, var, function, params, support=None, label=None, var_label=None):
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
        self.var_label = var_label # the label of the linguistic term's corresponding variable
    def __str__(self):
        return '%s IS %s' % (self.var_label, self.label)
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
    def find(self, label):
        """ Find the linguistic term that matches the provided label. """
        if '+' in label:
            label = label.split(' + ')[0]
        for term in self.terms:
            if term.label == label:
                return term
        return None
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
        self.weight = 1.0
#        self.weight = np.random.random(1)[0]
        self.active = True
    def __str__(self):
        text = 'IF '
        for i in range(len(self.antecedents)):
            antecedent = self.antecedents[i]
            text += str(antecedent)
            if i + 1 == len(self.antecedents):
                text += ', THEN '
            else:
                text += ', AND ' 
        for i in range(len(self.consequents)):
            consequent = self.consequents[i]
            text += str(consequent)
            if i + 1 < len(self.consequents):
                text += ', '
        return text
    def degreeOfApplicability(self, k, inpts):
        """ Determines the degree of applicability for this rule. """
        mus = []
        for idx in range(len(inpts)):
            antecedent = self.antecedents[idx]
            if antecedent != None:
                mu = antecedent.degree(inpts[idx])
                mus.append(mu)
        return min(mus)
    def lst(self):
        return self.antecedents + self.consequents
    
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
    def action(self, t, explore):
        F = self.asn.forward(t)
        self.Fs.append(F)
        F_prime = self.sam.F_prime(t)
        if not(explore):
            F_prime = F
        self.F_primes.append(F_prime)
        return F_prime

class GenericASN():
    """ Generic Action Selection Network """
    def __init__(self, X, Fs, R_hat, inputVariables, outputVariables, rules):
        self.X = X
        self.Fs = Fs
        self.R_hat = R_hat
        self.k = 10 # the degree of hardness for softmin
#        self.eta = 0.15 # the learning rate
#        self.eta = 0.0000000005 # achieved 186 reward
#        self.eta = 75e-12
#        self.eta = 25e-12
        self.eta = 5e-7
        self.inputVariables = inputVariables
        self.outputVariables = outputVariables
        self.antecedents = self.__antecedents() # generates the antecedents layer
        self.o1o2Weights = self.__o2() # assigns the weights between input layer and terms layer
        self.rules = self.__activeRules(rules) # generates the rules layer
        self.o2o3Weights = self.__o3() # assigns the weights between antecedents layer and rules layer
        self.consequents = self.__consequents() # generates the consequents layer
        self.o3o4Weights = self.__o4() # assigns the weights between rules layer and consequents layer
        self.O4 = {}
    def __str__(self):
        text = 'There are %s rules in the action selection network:\n\n' % (len(self.rules))
        for i in range(len(self.rules)):
            rule = self.rules[i]
            text += '%s. ' % (i + 1)
            text += str(rule)
            if i + 1 != len(self.rules):
                text += '\n\n'
        return text
    def __activeRules(self, rules):
        active_rules = []
        for rule in rules:
            if rule.active:
                active_rules.append(rule)
        return active_rules
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
#                if consequent.label in rule.consequents[0].label:
                    weights[row, col] = 1
        return weights            
    def updateRules(self, new_rules):
        """ Change/update the rules used in the action selection network. """
        self.rules = new_rules # generates the rules layer
        self.o2o3Weights = self.__o3() # assigns the weights between antecedents layer and rules layer
        self.consequents = self.__consequents() # generates the consequents layer
        self.o3o4Weights = self.__o4() # assigns the weights between rules layer and consequents layer
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
        if (self.Fs[t] - self.Fs[t-1]) == 0: # divide by zero error is possible here, investigate why later
            dv_dF = (aen.v(t, t) - aen.v(t-1, t-1))
        else:
            dv_dF = (aen.v(t, t) - aen.v(t-1, t-1)) / (self.Fs[t] - self.Fs[t-1])
        numerator = 0.0
        denominator = 0.0
        
        o4activation = self.O4[t]
        
        for idx in range(len(self.consequents)):
            u_i = o4activation[idx]
            consequent = self.consequents[idx]
            numerator += consequent.params['center'] * consequent.params['sigma'] * u_i
            denominator += consequent.params['sigma'] * u_i
        
#        print('dv_dF: %s' % dv_dF)
#        print('numerator: %s' % numerator)
#        print('denominator: %s' % denominator)
            
        for idx in range(len(self.consequents)):
            u_i = o4activation[idx]
            consequent = self.consequents[idx]
#            consequent.params['center'] += self.eta * dv_dF * ((consequent.params['sigma'] * u_i) / denominator)
#            consequent.params['sigma'] += self.eta * dv_dF * (((consequent.params['center'] * u_i * denominator) - (numerator * u_i)) / (pow(denominator, 2)))
            consequent.params['center'] += self.eta * np.sign(dv_dF) * ((consequent.params['sigma'] * u_i) / denominator)
#            consequent.params['sigma'] += self.eta * (((consequent.params['center'] * u_i * denominator) - (numerator * u_i)) / (pow(denominator, 2))) # remove dv_dF because it results in sigma becoming negative, which makes no sense

#            consequent.params['sigma'] += self.eta * np.sign(dv_dF) * (((consequent.params['center'] * u_i * denominator) - (numerator * u_i)) / (pow(denominator, 2))) # can result in large negative numbers -- need to fix

        # (2/2) tune antecedents
        
        delta_5 = 1
#        delta_5 = np.sign(dv_dF)
        delta_4 = {} # indexed by consequents
        for idx in range(len(self.consequents)):
            u_i = o4activation[idx]
            consequent = self.consequents[idx]
            # correct, but causes large numbers
            delta_4[idx] = delta_5 * (((consequent.params['center'] * consequent.params['sigma'] * denominator) - (numerator * consequent.params['sigma'])) / (pow(denominator, 2)))
#            delta_4[idx] = delta_5 * (((consequent.params['center'] * consequent.params['sigma'] * denominator) - (numerator * consequent.params['sigma'])) / (pow(denominator, 1)))

        delta_3 = delta_4
                
        for i in range(len(self.X[t])):
            x = self.X[t]
            u_i = x[i]
            o2 = np.where(self.o1o2Weights[i]==1.0)[0] # get all indexes that this input maps to
            for j in o2:
                antecedent = self.antecedents[j]
                a_i = antecedent.degree(u_i) # degree currently includes e^f, so this is actually activation function
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
                        # find the error of this rule's consequence
                        
                        # find the consequent's index of this rule
                        consequentIndexes = np.where(self.o3o4Weights[k]==1.0)[0]
                        
                        # the error of the kth rule is the summation of the errors of its consequences
                        for consequentIndex in consequentIndexes:
                            q_k += delta_3[consequentIndex]
                    else:
                        q_k = 0.0 # do nothing
                    
                    dE_da_i += q_k
                    
                    # --- TRYING SOMETHING NEW HERE ---
        
#                    rule.weight = np.tanh(rule.weight + (self.eta * dE_da_i))
#                    
#                    if rule.weight >= 0:
##                        if not(rule.active):
##                            print('%sth rule dropping in' % k)
#                        rule.active = True
#                    else:
##                        if rule.active:
##                            print('%sth rule dropping out' % k)
#                        rule.active = False
                    
                    # --- END OF TRYING SOMETHING NEW HERE ---
                
                # UPDATE THE WEIGHTS
#                print('dE_da_i %s' % dE_da_i)
#                print('delta_m_ij %s' % delta_m_ij)
#                print('change of %s center = %s' % (antecedent.label, self.eta * dE_da_i * delta_m_ij))
                antecedent.params['center'] += self.eta * dE_da_i * delta_m_ij
#                antecedent.params['sigma'] += self.eta * dE_da_i * delta_sigma_ij
#                print('c = %s , sigma = %s' % (antecedent.params['center'], antecedent.params['sigma']))