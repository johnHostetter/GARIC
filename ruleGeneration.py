# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 19:27:59 2020

@author: jhost
"""

import math
import copy
import numpy as np
from neurofuzzynetwork import Rule

np.random.seed(0)

def fuzzify(NFN_variables, x):
    """ Fuzzify an observation to its corresponding fuzzy regions. """
    new_x = {'str':[], 'obj':[]}
    for inpt_idx in range(len(x)):
        inpt = x[inpt_idx]
        max_term = None
        max_deg = 0
        for term in NFN_variables[inpt_idx].terms:
            deg = term.degree(inpt)
            if deg > max_deg:
                max_deg = deg
                max_term = term
        # AttributeError: 'NoneType' object has no attribute 'label'
        new_x['str'].append(max_term.label + ' + ' + NFN_variables[inpt_idx].label)
        new_x['obj'].append(max_term)
    return new_x

def rule_deg(fuzzy_observation, X, V):
    """ Determine the degree of a potential rule. """
    
    # following performs quite well alone - achieved 125 reward and mean 36
    
    k = 1.0 # power of significance --- influences weight of rule value, 0.5 to 0.75 worked well
    summ = 0.0
    deg = 1.0
    for i in range(len(X)):
        x = X[i]
        v = V[i]
        for idx in range(len(fuzzy_observation[:4])):
            deg *= fuzzy_observation[idx].degree(x[idx])
#        norm_v = v
        norm_v = np.tanh(pow(v,k)) # the tanh with pow seemed to outperform reg. normalization
#        norm_v = v / max(V)
        summ += (deg * norm_v)
    return summ / len(X)

def make_rules_dictionary(NFN_Variables, X, V):
    rules_dictionary = {}
    for i in range(len(X)):
        x = X[i]
        v = V[i]
        new_x = fuzzify(NFN_Variables, x)
        key = ' - '.join(new_x['str'][:4])
        value = ' - '.join(new_x['str'][4:])
        try:
            rules_dictionary[key]['set'].add(value)
            rules_dictionary[key]['X'].append(x)
            rules_dictionary[key]['V'].append(v)
        except KeyError:
            rules_dictionary[key] = {'set':set(), 'X':[], 'V':[]}
            rules_dictionary[key]['set'].add(value)
            rules_dictionary[key]['X'].append(x)
            rules_dictionary[key]['V'].append(v)
    return rules_dictionary

def filter_rules(NFN_variables, rules_dictionary, X):
    
    # to determine the threshold, perform the following:
    max_degs = []
    for key in rules_dictionary.keys():
        antecedents = key.split(' - ')
        max_rule = None
        max_rule_deg = 0.0
        consequents = set(rules_dictionary[key]['set'])
        for consequent in consequents:
            str_rule = copy.deepcopy(antecedents)
            str_rule.append(consequent)
            rule = []
            var_idx = 0
            for str_term in str_rule:
                rule.append(NFN_variables[var_idx].find(str_term))
                var_idx += 1
            deg = rule_deg(rule, rules_dictionary[key]['X'], rules_dictionary[key]['V'])
            if deg > max_rule_deg:
                max_rule_deg = deg
                max_rule = rule
        max_degs.append(max_rule_deg)
    
    
    
#     determining threshold size
    max_degs = sorted(max_degs)
    n = math.floor(len(max_degs)/4)
    print('potential rules: %s' % len(max_degs))
    print('set: %s' % len(set(max_degs)))
    print('max: %s' % max(set(max_degs)))
#    threshold = max(max_degs) * 0.5
    max_degs = max_degs[-n:]
    threshold = min(max_degs)
    print('threshold: %s' % threshold) 
    
    
#    threshold = 0.1 # a parameter, 0.1 worked well and got 125 max reward
    fuzzy_rules = []
    for key in rules_dictionary.keys():
        antecedents = key.split(' - ')
        max_rule = None
        max_rule_deg = 0.0
        consequents = set(rules_dictionary[key]['set'])
        for consequent in consequents:
            str_rule = copy.deepcopy(antecedents)
            str_rule.append(consequent)
            rule = []
            var_idx = 0
            for str_term in str_rule:
                rule.append(NFN_variables[var_idx].find(str_term))
                var_idx += 1
            deg = rule_deg(rule, rules_dictionary[key]['X'], rules_dictionary[key]['V'])
            if deg > max_rule_deg:
                max_rule_deg = deg
                max_rule = rule
        if max_rule_deg > threshold:
            fuzzy_rules.append(max_rule)
    return fuzzy_rules

def create_rules(fuzzy_rules):
    rules = []
    for fuzzy_rule in fuzzy_rules:
        rules.append(Rule(fuzzy_rule[:4], fuzzy_rule[4:]))
    return rules

def update_rules(NFN_variables, X, V):
    rules_dictionary = make_rules_dictionary(NFN_variables, X, V)
    fuzzy_rules = filter_rules(NFN_variables, rules_dictionary, X)
    return create_rules(fuzzy_rules)[0:25]

def init_rules(NFN_variables, X, V):
    rules_dictionary = make_rules_dictionary(NFN_variables, X, V)
    rules = create_rules(filter_rules(NFN_variables, rules_dictionary, X))
    return rules