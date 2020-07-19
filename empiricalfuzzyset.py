# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:52:22 2020

@author: jhost
"""

import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from collections import Counter
from statistics import stdev 
from continuous_cartpole import ContinuousCartPoleEnv

from neurofuzzynetwork import Term, Variable, Rule, eFL_ACC
from garic import AEN

random.seed(0)

numerator = None
distances = powers = np.array([])

def unimodalDensity(X, i, dist):
    """ Calculate the unimodal density of a particular ith data sample
    from the set of observations, X, with the distance metric, dist. """
    global numerator, distances, powers
    K = len(X)
#    numerator = 0
    denominator = 0
    
    if len(distances) == 0:
        print('Building matrices...')
        distances = np.array([float('inf')]*len(X)*len(X)).reshape(len(X), len(X))
        powers = np.array([float('inf')]*len(X)*len(X)).reshape(len(X), len(X))
        print('Done.')
        print('Standby for initialization...')
    
    if numerator == None:
        numerator = 0
        for k in range(K):
            for j in range(K):
                numerator += pow(dist(X[k], X[j]), 2)
    for j in range(K):
        if distances[i, j] != float('inf') or distances[j, i] != float('inf'):
            denominator += distances[i, j]
        else:    
            distances[i, j] = dist(X[i], X[j])
            powers[i, j] = pow(distances[i, j], 2)
            distances[j, i] = distances[i, j]
            powers[j, i] = powers[i, j]
            denominator += powers[i, j]
        
    denominator *= 2 * K
    return numerator / denominator

def unimodalDensity1(X, x, dist):
    """ Calculate the unimodal density of a particular ith data sample
    from the set of observations, X, with the distance metric, dist. """
    global numerator, distances, powers
    K = len(X)
#    numerator = 0
    denominator = 0
    
    if len(distances) == 0:
        print('Building matrices...')
        distances = np.array([float('inf')]*len(X)*len(X)).reshape(len(X), len(X))
        powers = np.array([float('inf')]*len(X)*len(X)).reshape(len(X), len(X))
        print('Done.')
        print('Standby for initialization...')
    
    if numerator == None:
        numerator = 0
        for k in range(K):
            for j in range(K):
                numerator += pow(dist(X[k], X[j]), 2)
    for j in range(K):
        denominator += pow(dist(x, X[j]), 2)
        
    denominator *= 2 * K
    return numerator / denominator

def multimodalDensity(X, U, F, i, dist):
    """ Calculate the multimodal density of a particular ith data sample
    from the set of observations, X, with the distance metric, dist, and
    using the set of frequencies, F. """
#    idx = X.index(U[i])
    idx = i
    return F[i] * unimodalDensity(X, idx, dist)

def multimodalDensity1(X, x, dist):
    """ Calculate the multimodal density of a particular ith data sample
    from the set of observations, X, with the distance metric, dist, and
    using the set of frequencies, F. """
#    idx = X.index(U[i])
    return unimodalDensity1(X, x, dist)

def unique(X):
    counter = Counter(X)
    U = list(counter.keys()) # unique observations
    F = list(counter.values()) # frequencies
    return (U, F)

def plotDistribution(X, Y, title):
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
#    plt.plot(X, Y, 'o', color='blue')
    plt.plot(X, Y, 'o', color='blue')
    plt.legend()

X = []

#env = gym.make("CartPole-v1")
env = ContinuousCartPoleEnv()
env.seed(13)
env.min_action = -1.0
env.max_action = 1.0
env.action_space = gym.spaces.Box(
    low=env.min_action,
    high=env.max_action,
    shape=(1,)
)

steps = [] # observations for the current episode
episodes = {}
time_to_episodes = {}
episode_ctr = 0 # counter for which episode the environment is currently on
total_reward = 0.0 # total reward for this episode so far
observation = env.reset()

reset = True
aen = AEN([], [], [], 4, h=5)
aen.X.append(observation) # GARIC observe the environment
aen.R.append(0)
aen.R_hat.append(0)

for t in range(300):
    env.render()
#    action = np.array([random.choice([-1,1])], dtype='float32')  
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
#    observation[0] = np.tanh(observation[0])
#    observation[1] = np.tanh(observation[1])
#    observation[3] = np.tanh(observation[3])
#    observation[2] = np.tanh(observation[2])
    # add force to observation
    observation = list(observation)
    observation.append(action[0])
    observation = np.array(observation)
#    print(observation)
    X.append(np.array(observation))
    
    # add to AEN
    
    aen.X.append(observation) # GARIC observe the environment
    aen.R.append(reward) # GARIC observe the reward
    aen.r_hat(t, done, reset) # GARIC determine internal reinforcement
    aen.backpropagation(t) # GARIC update weights on AEN
    
    # add to episode history
    
    total_reward += reward
    steps.append(observation)
    time_to_episodes[t] = episode_ctr
    
    reset = False
    if done:
        avg_reward = total_reward / len(steps)
        episodes[episode_ctr] = {'steps':steps, 'value':avg_reward}
        observation = env.reset()
        total_reward = 0.0
        episode_ctr += 1
        steps = []
        reset = True
        
if not(done):
    avg_reward = total_reward / len(steps)
    episodes[episode_ctr] = {'steps':steps, 'value':avg_reward}
    observation = env.reset()
    total_reward = 0.0
    episode_ctr += 1
    steps = []

env.close()

#X.sort()
#U, F = unique(X)
U = X
F = np.array([1]*len(X))

x_lst = []
y_lst = []
densities = {}

# step 1

for i in range(len(U)):
    mm = multimodalDensity(X, U, F, i, distance.euclidean)
    print('%s/%s: %s' % (i, len(U), mm))
    densities[mm] = i # add the multimodal density to the set
    x_lst.append(U[i])
    y_lst.append(mm)
    
# step 2
    
maximum_multimodal_density = max(densities.keys()) # find the maximum multimodal density
idx = densities[maximum_multimodal_density] # find the index of the unique data sample with the maximum multimodal density
u1_star = U[idx] # find the unique data sample with the maximum multimodal density

# step 3

#U.pop(idx) # remove from the set of unique observations
DMM = []
ULstar = []
uR = u1_star

ctr = 0
visited = {}
previousLeftIndex = 0
previousLeftValue = 0
localMaximaIndexes = []
prototypes = []
while len(visited.keys()) < len(U):
    # step 4
    srted = sorted(distances[idx])
    for uRidx in range(len(srted)):
        item = srted[uRidx]
        if uRidx == idx:
            continue
        if uRidx in visited.keys():
            continue
        else:
            visited[uRidx] = ctr
            ctr += 1
            ULstar.append(U[uRidx])
            idx = uRidx # step 5, now go back to step 4
            mm = multimodalDensity(X, U, F, uRidx, distance.euclidean)
            DMM.append(mm) # step 6
            # step 7 is both if & else statements
            if mm > previousLeftValue:
                previousLeftIndex = uRidx
                previousLeftValue = mm
            else:
                localMaximaIndexes.append(uRidx - 1)
                prototypes.append(U[uRidx - 1])
                previousLeftIndex = uRidx
                previousLeftValue = mm
            break
#    row = idx
#    col = 0
#    U = np.delete(U, row, col)

# step 8
clouds = {} # each element is a list that is indexed by a prototype index
labels = [] # a direct labeling where the ith element of X has an ith label
for x in X:
    min_p = None
    min_idx = float('inf')
    min_dist = float('inf')
    for prototype_idx in range(len(prototypes)):
        prototype = prototypes[prototype_idx]
        dist = distance.euclidean(x, prototype)
        if dist < min_dist:
            min_p = prototype
            min_idx = prototype_idx
            min_dist = dist
    labels.append(min_p)
    try:
        clouds[min_idx].append(x)
    except KeyError:
        clouds[min_idx] = []
        clouds[min_idx].append(x)
#    print(min_p)

# step 9
p0 = {} # the centers of the prototypes
for prototype_idx in clouds.keys():
    elements = clouds[prototype_idx]
    center = sum(elements) / len(elements)
    p0[prototype_idx] = center

# step 10
dmm_p0 = {}
for prototype_idx in p0.keys():
    dmm_p0[prototype_idx] = multimodalDensity1(X, p0[prototype_idx], distance.euclidean)
    
print('Calculating multimodal densities of prototypes and reducing number of prototypes...')
    
# step 11
runAgain = True
iteration = 0
while(runAgain):
    n = 0 # the number of unique pairs
    eta = 0.0
    ds = []
    sigma = 0.0
    for i in p0.keys():
        for j in p0.keys():
            if i > j:
                d = distance.euclidean(p0[i], p0[j])
                ds.append(d)
                eta += d
                n += 1
    eta /= n
    sigma = stdev(ds)
    R = sigma * (1 - (sigma / eta))
    piN = {}
    
    for i in p0.keys():
        for j in p0.keys():
            d = distance.euclidean(p0[j], p0[i])
            if d < R:
                try:
                    piN[i].append(p0[j])
                except KeyError:
                    piN[i] = []
                    piN[i].append(p0[j])
    
    p1 = {}
    for i in p0.keys():
        pi = p0[i]
        max_val = dmm_p0[i]
        member = True
        for q in piN[i]:
            if multimodalDensity1(X, q, distance.euclidean) > max_val:
                member = False
        if member:
            p1[i] = pi
            
    print(iteration)
    iteration += 1
    print('%s vs %s' % (len(p1.keys()), len(p0.keys())))
    runAgain = len(p1.keys()) < len(p0.keys()) # step 13
    p0 = p1 # step 12
    
print('Calculating final prototypes and creating data clouds...')

# step 14
clouds = {} # each element is a list that is indexed by a prototype index
labels = [] # a direct labeling where the ith element of X has an ith label
for x in X:
    min_p = None
    min_idx = float('inf')
    min_dist = float('inf')
    for prototype_idx in p0.keys():
        prototype = p0[prototype_idx]
        dist = distance.euclidean(x, prototype)
        if dist < min_dist:
            min_p = prototype
            min_idx = prototype_idx
            min_dist = dist
    labels.append(min_p)
    try:
        clouds[min_idx].append(x)
    except KeyError:
        clouds[min_idx] = []
        clouds[min_idx].append(x)
    
def gaussianMembership(x, center, sigma):
    numerator = (-1) * pow(x - center, 2)
    denominator = 2 * pow(sigma, 2)
    return pow(math.e, numerator / denominator)

def mystdev(lst, i):
    """ Calculate the standard deviation of the ith feature
    from a list of observations that have been collected. """
    new_lst = []
    for item in lst:
        new_lst.append(item[i])
    return stdev(new_lst)

def distMatrix(terms):
    matrix = np.array([float('inf')]*len(terms)*len(terms)).reshape(len(terms), len(terms))
    for i in terms.keys():
        for j in terms.keys():
            i_matrix = np.where(np.array(list(terms.keys()))==i)[0][0]
            j_matrix = np.where(np.array(list(terms.keys()))==j)[0][0]
            matrix[i_matrix, j_matrix] = distance.euclidean(terms[i]['center'], terms[j]['center'])
    return matrix

def identifySimilarPair(terms, distMatrix):
    min_i = -1
    min_j = -1
    min_dist = float('inf')
    for i in range(len(distMatrix)):
        for jt in range(len(distMatrix[0])):
            if i < jt:
                d = distMatrix[i, jt]
                if d < min_dist:
                    min_dist = d
                    min_i = i
                    min_j = jt
    min_i_key = list(terms.keys())[min_i]
    min_j_key = list(terms.keys())[min_j]
    return {'pair':(min_i_key, min_j_key), 'distance':min_dist}

new_p_idxs = -1
def reduction(clouds, terms, similarity):
    global new_p_idxs
    tple = similarity['pair']
    if not(tple[0] == -1 and tple[1] == -1): # nothing found
        A = tple[0]
        B = tple[1]
        c_A = terms[A]['center']
        c_B = terms[B]['center']
        support_A = terms[A]['support']
        support_B = terms[B]['support']
        support_C = support_A + support_B
        c_new = (support_A/support_C) * c_A + (support_B/support_C) * c_B
        clouds_new = clouds[terms[A]['p_idx']]
        clouds_new.extend(clouds[terms[B]['p_idx']])
        clouds[new_p_idxs] = clouds_new
        sig = mystdev(clouds_new, feature_idx)
    #    clouds.pop(terms[A]['p_idx'])
    #    clouds.pop(terms[B]['p_idx'])
        del terms[A]
        del terms[B]
        term_new = {'center':c_new, 'sigma':sig, 'support':support_C, 'p_idx':new_p_idxs}
        terms[new_p_idxs] = term_new
        new_p_idxs -= 1
        return term_new
    return None

def compression(clouds, threshold, num_of_terms, terms, feature_idx):
    matrix = distMatrix(terms[feature_idx])
    similarity = identifySimilarPair(terms[feature_idx], matrix)
    print('num of terms remaining: %s' % len(terms[feature_idx]))
    if len(terms[feature_idx]) > num_of_terms or similarity['distance'] < threshold:
        result = reduction(clouds, terms[feature_idx], similarity)
        if result == None:
            return False # stop, do not continue
        else:
            return True # continue
    return False # stop, do not continue
    
variables = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}}
features = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Velocity At Tip', 'Force']
for feature_idx in range(len(features)):
    x_lst = []
    mu_lst = []
    for p_idx in p0.keys():
        if len(clouds[p_idx]) > 1:
            c = p0[p_idx][feature_idx]
            sig = mystdev(clouds[p_idx], feature_idx)
            variables[feature_idx][p_idx] = {'center':c, 'sigma':sig, 'support':len(clouds[p_idx]), 'p_idx':p_idx}
            for x in X:
                x = x[feature_idx]
                mu = gaussianMembership(x, c, sig)
                x_lst.append(x)
                mu_lst.append(mu)
            title = features[feature_idx]
#            plotDistribution(x_lst, mu_lst, title)
        plt.show()

print('--- COMPRESSING LINGUISTIC TERMS ---')

thresholds = [0.01, 0.01, 0.001, 0.01, 0.01] # the corresponding threshold for each feature
#thresholds = [0.1, 0.1, 0.001, 0.1, 0.1] # the corresponding threshold for each feature
for feature_idx in range(len(features)):
    cont = True
    while cont:
        if feature_idx != 4:
            cont = compression(clouds, thresholds[feature_idx], 15, variables, feature_idx)
        else:
            cont = compression(clouds, thresholds[feature_idx], 15, variables, feature_idx)

features = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Velocity At Tip', 'Force']
for feature_idx in range(len(features)):
    x_lst = []
    mu_lst = []
    for p_idx in variables[feature_idx].keys():
        c = variables[feature_idx][p_idx]['center']
        sig = variables[feature_idx][p_idx]['sigma']
        for x in X:
            x = x[feature_idx]
            mu = gaussianMembership(x, c, sig)
            x_lst.append(x)
            mu_lst.append(mu)
        title = features[feature_idx]
        plotDistribution(x_lst, mu_lst, title)
    plt.show()
        
# trying to incorporate empirical fuzzy sets into neuro fuzzy networks

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


NFN_variables = [] # variables meant for the neuro fuzzy network
term_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
all_terms = []
for var_key in variables.keys():
    idx = var_key
    var_label = features[var_key]
    terms = []
    term_label_idx = 0
#    index = 0
#    min_c_idx = 0
#    max_c_idx = 0
#    min_c = float('inf')
#    max_c = -float('inf')
#    for term_key in variables[var_key].keys():
#        c = variables[var_key][term_key]['center']
#        if c < min_c:
#            min_c = c
#            min_c_idx = index
#        if c > max_c:
#            max_c = c
#            max_c_idx = index
#        index +=1
    for term_key in variables[var_key].keys():
        c = variables[var_key][term_key]['center']
        sig = variables[var_key][term_key]['sigma']
        sup = variables[var_key][term_key]['support']
        params = {'center':c, 'sigma':sig}
        term_label = term_labels[term_label_idx]
#        if term_label_idx == min_c_idx:
#            term = Term(var_key, NFN_rightSigmoidMembership, params, sup, term_label)
#        elif term_label_idx == max_c_idx:
#            term = Term(var_key, NFN_leftSigmoidMembership, params, sup, term_label)
#        else:    
#            term = Term(var_key, NFN_bellShapedMembership, params, sup, term_label)
        
        term = Term(var_key, NFN_bellShapedMembership, params, sup, term_label)
        
        term_label_idx += 1
        terms.append(term)
    all_terms.append(terms)
    variable = Variable(idx, terms, var_label)
    NFN_variables.append(variable)

# --- TRYING SOMETHING NEW ---
    
# trying to find the rules
    
#import itertools

#rule_antecedent_combinations = list(itertools.product(*all_terms))

#new_X = []
#
#for x in X:
#    new_x = []
#    for inpt_idx in range(len(x)):
#        inpt = x[inpt_idx]
#        max_term = None
#        max_deg = 0
#        for term in NFN_variables[inpt_idx].terms:
#            deg = term.degree(inpt)
#            if deg > max_deg:
#                max_deg = deg
#                max_term = term
#        new_x.append(max_term.label + ' + ' + NFN_variables[inpt_idx].label)
#    new_X.append(new_x)
#
#rules = []
##for antecedent_combination in rule_antecedent_combinations:
##    for consequent_term in NFN_variables[4].terms:
##        rules.append(Rule(antecedent_combination, [consequent_term]))
#        
## build a weights matrix for the links between the rule nodes and consequent nodes
#weights = np.array([1.0]*(len(rule_antecedent_combinations)*len(NFN_variables[4].terms))).reshape(len(rule_antecedent_combinations), len(NFN_variables[4].terms))
#    
## competitive learning
#for idx in range(len(X)):
#    x = X[idx]
#    print('%s / %s' % (idx, len(X)))
#    for i in range(len(rule_antecedent_combinations)):
#        for jidx in range(len(NFN_variables[4].terms)):
#            o3_i = Rule(rule_antecedent_combinations[i],[]).degreeOfApplicability(0, x[:4])
#            o4_j = NFN_variables[4].terms[jidx].degree(x[4])
##            print('o3 = %s, o4 = %s' % (o3_i, o4_j))
#            weights[i,jidx] += (o3_i * o4_j)
#          
#test = {0:0.0, 1:0.0, 2:0.0, 3:0.0}
#rules_dict = {0:[], 1:[], 2:[], 3:[]}
#for idx in range(len(rule_antecedent_combinations)):
#    val = max(weights[idx])
#    index = np.where(weights[idx]==val)[0][0]
#    rule = Rule(rule_antecedent_combinations[idx], [NFN_variables[4].terms[index]])
#    rules.append(rule)
#    rules_dict[index].append(rule)
#    test[index] += 1
#    
#print(test)
#
#final_rules = []
#
#for i in rules_dict.keys():
#    print(i)
#    # get top 10 rules
#    for _ in range(10):
#        try:
#            importance = [0.0]*(len(rules_dict[i]))
#            for k in range(len(rules_dict[i])):
#                for idx in range(len(X)):
#                    x = X[idx]
#                    importance[k] += rules_dict[i][k].degreeOfApplicability(0, x[:4]) * rules_dict[i][k].consequents[0].degree(x[4])
#            idx = importance.index(max(importance))
#            final_rules.append(rules_dict[i][idx])
#            rules_dict[i].pop(idx)
#        except ValueError:
#            break
# --- END OF TRYING SOMETHING NEW ---

import pandas as pd
from apyori import apriori

print('Transforming observations to their fuzzy counterparts...')

def fuzzify(x):
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
        new_x['str'].append(max_term.label + ' + ' + NFN_variables[inpt_idx].label)
        new_x['obj'].append(max_term)
    return new_x

def rule_deg(fuzzy_observation, X):
    """ Determine the degree of a potential rule. """
    summ = 0.0
    deg = 1.0
#    for idx in range(len(fuzzy_observation)):
#        deg *= fuzzy_observation[idx].degree(x[idx])
#    return deg
#    terms = fuzzy_observation['obj']
    for x in X:
        for idx in range(len(fuzzy_observation)):
            deg *= fuzzy_observation[idx].degree(x[idx])
        summ += deg
    return summ / len(X)

def make_rules_dictionary(X, new_X, fuzzy_X):
    rules_dictionary = {}
    for x in X:
        new_x = fuzzify(x)
        new_X.append(new_x['str'])
        fuzzy_X.append(new_x)
        key = ' - '.join(new_x['str'][:4])
        value = ' - '.join(new_x['str'][4:])
        try:
            rules_dictionary[key]['set'].add(value)
            rules_dictionary[key]['X'].append(x)
        except KeyError:
            rules_dictionary[key] = {'set':set(), 'X':[]}
            rules_dictionary[key]['set'].add(value)
            rules_dictionary[key]['X'].append(x)
    return rules_dictionary
new_X = []
fuzzy_X = []
rules_dictionary = make_rules_dictionary(X, new_X, fuzzy_X)
#for x in X:
#    new_x = fuzzify(x)
#    new_X.append(new_x['str'])
#    fuzzy_X.append(new_x)
#    key = ' - '.join(new_x['str'][:4])
#    value = ' - '.join(new_x['str'][4:])
#    try:
#        rules_dictionary[key].append(value)
#    except KeyError:
#        rules_dictionary[key] = []
#        rules_dictionary[key].append(value)
        
import copy

def filter_rules(rules_dictionary, X):
    fuzzy_rules = []
    for key in rules_dictionary.keys():
        antecedents = key.split(' - ')
        max_rule = None
        max_rule_deg = 0.0
        consequents = set(rules_dictionary[key]['set'])
        for consequent in consequents:
#                consequents = rules_dictionary[key].split(' - ')
            str_rule = copy.deepcopy(antecedents)
            str_rule.append(consequent)
            rule = []
            var_idx = 0
            for str_term in str_rule:
                rule.append(NFN_variables[var_idx].find(str_term))
                var_idx += 1
            deg = rule_deg(rule, rules_dictionary[key]['X'])
            if deg > max_rule_deg:
                if max_rule_deg > 0:
                    print('overwritten')
                max_rule_deg = deg
                max_rule = rule
        fuzzy_rules.append(max_rule)
    return fuzzy_rules

def create_rules(fuzzy_rules):
    rules = []
    for fuzzy_rule in fuzzy_rules:
        rules.append(Rule(fuzzy_rule[:4], fuzzy_rule[4:]))
    return rules

def update_rules(X):
    rules_dictionary = make_rules_dictionary(X, [], [])
    fuzzy_rules = filter_rules(rules_dictionary, X)
    return create_rules(fuzzy_rules)

rules = create_rules(filter_rules(rules_dictionary, X))
    
#association_rules = apriori(new_X, min_support=0.005, min_confidence=0.2, min_lift=2, min_length=2)
#association_results = list(association_rules)
#
#results = []
#for item in association_results:
#    lhs = " - ".join(list(item[2][0].items_base))
#    rhs = " - ".join(list(item[2][0].items_add))
#    support = item.support
#    freq = support * len(association_results)
#    confidence = item[2][0].confidence
#    lift = item[2][0].lift
#    rows = (lhs, rhs, support, confidence, lift, freq)
#    results.append(rows)
#
#labels = ['LHS','RHS','Support','Confidence','Lift', 'Frequency']
#rules_out = pd.DataFrame.from_records(results, columns = labels)
#
#force_rules = rules_out[rules_out['RHS'].str.contains("Force")]
#force_rules = force_rules.sort_values(['Support', 'Confidence', 'Lift'], ascending=False) # sort first by confidence, then by lift
#
## organize the rules information to find the most prevalent rule antecedents
#
#print('Finding the most prevalent associations...')
#
#lhs_to_rhs = {}
#rhs_to_lhs = {}
#for idx in force_rules.index:
#    generated_rule = force_rules.loc[idx]
#    if len(generated_rule['LHS']) > 0:
#        rhs_list = generated_rule['RHS'].split(' - ')
#        res = [i for i in rhs_list if 'Force' in i][0]
#        try:
##            rhs_to_lhs[res].add(generated_rule['LHS'])
#            if len(rhs_to_lhs[res]) < 3:
##            if len(rhs_to_lhs[res]) < 5:
#                rhs_to_lhs[res].add(generated_rule['LHS'])
#            else:
#                continue
#        except KeyError:
#            rhs_to_lhs[res] = set()
#            rhs_to_lhs[res].add(generated_rule['LHS'])
#        try:
#            lhs_to_rhs[str(generated_rule['LHS'])]
#        except KeyError:
#            lhs_to_rhs[str(generated_rule['LHS'])] = res
#            
## create the rules for the neuro fuzzy network   
#
#print('Creating fuzzy logic control rules...')
#
#rules = []
#for lhs in lhs_to_rhs.keys():
#    lhs_terms = lhs.split(' - ')
#    rhs_terms = lhs_to_rhs[lhs].split(' - ')
#    antecedents = []
#    consequents = []
#    for lhs_term in lhs_terms:
#        tokens = lhs_term.split(' + ')
#        variable = tokens[1]
#        term = tokens[0]
#        # find the corresponding variable and term
#        for NFN_variable in NFN_variables[0:4]:
#            if NFN_variable.label == variable:
#                for NFN_term in NFN_variable.terms:
#                    if NFN_term.label == term:
#                        antecedents.append(NFN_term)
#            else:
#                antecedents.append(None)
#    for rhs_term in rhs_terms:
#        tokens = rhs_term.split(' + ')
#        variable = tokens[1]
#        term = tokens[0]
#        # find the corresponding variable and term
#        for NFN_variable in NFN_variables[4:]:
#            if NFN_variable.label == variable:
#                for NFN_term in NFN_variable.terms:
#                    if NFN_term.label == term:
#                        consequents.append(NFN_term)
#            else:
#                consequents.append(None)
#    rules.append(Rule(antecedents, consequents))
#
## find better rule consequents
#
#print('Modifying rule consequents...')
#
#rule_idx = 0
#for lhs in lhs_to_rhs.keys():
##    print(rule_idx)
#    lhs_terms = lhs.split(' - ')
#    rhs_terms = lhs_to_rhs[lhs].split(' - ')
#    antecedents = []
#    poss_consequents = {}
#    for t in range(len(new_X)):
#        x = X[t]
#        new_x = new_X[t]
#        if set(lhs_terms).issubset(new_x):
#            episode_ctr = time_to_episodes[t]
#            # find the index of the observation in the episode
#            for idx in range(len(episodes[episode_ctr]['steps'])):
#                if (episodes[episode_ctr]['steps'][idx] == x).all():  
#                    relative_time = 1 - (idx / len(episodes[episode_ctr]['steps']))
#                    action = new_x[4]
#                    try:
#                        poss_consequents[action] += relative_time * episodes[episode_ctr]['value']
#                    except KeyError:
#                        poss_consequents[action] = 0
#                        poss_consequents[action] += relative_time * episodes[episode_ctr]['value']
#                    break
#    max_action_val = 0.0
#    max_action_term = None
#    for consequent_key in poss_consequents.keys():
#        if poss_consequents[consequent_key] > max_action_val:
#            max_action_val = poss_consequents[consequent_key]
#            max_action_term = consequent_key
#       
#    consequents = []
#    tokens = max_action_term.split(' + ')
#    variable = tokens[1]
#    term = tokens[0]
#    # find the corresponding variable and term
#    for NFN_variable in NFN_variables[4:]:
#        if NFN_variable.label == variable:
#            for NFN_term in NFN_variable.terms:
#                if NFN_term.label == term:
#                    consequents.append(NFN_term)
#        else:
#            consequents.append(None)
#    rules[rule_idx].consequents = consequents
#    rule_idx += 1
    
agent = eFL_ACC(NFN_variables[0:4], NFN_variables[4:], rules, 5, lower=-2, upper=2)

#env = gym.make("CartPole-v1")
env = ContinuousCartPoleEnv()
env.min_action = -100
env.max_action = 100
env.seed(20)
env.action_space = gym.spaces.Box(
    low=env.min_action,
    high=env.max_action,
    shape=(1,)
)


agent.aen = aen # testing loading up a pre-trained AEN
aen.X = agent.X
aen.R = agent.R
aen.R_hat = agent.R_hat



observation = env.reset()
agent.X.append(observation) # GARIC observe the environment
agent.R.append(0)
agent.R_hat.append(0)



t = 0
time_up = 0
total_r = 0
episode = 0
reset = True
for idx in range(5):
    NFN_variables[idx].graph(-1,1)
    
print('--- Demo of exploration and fine tuning of membership functions ---')
print('continue?')
yes = input()

observations = []
    
print(observation)
rewards = []
for _ in range(5000):
    env.render()
    
    if episode % 20 == 0 and reset and t <= 4000:
        print('Updating fuzzy logic control rules...')
        total_X = copy.deepcopy(X)
        total_X.extend(observations)
        rules = update_rules(total_X[-600:])
        agent.asn.updateRules(rules)
#        agent = eFL_ACC(NFN_variables[0:4], NFN_variables[4:], rules, 5, lower=-2, upper=2)
#        agent.X.append(observation) # GARIC observe the environment
#        agent.R.append(0)
#        agent.R_hat.append(0)
#        t = 0

    
    
#    if t > 100 and t % 200 == 0:
#        NFN_variables[4].graph(min(agent.F_primes) - 1, max(agent.F_primes) + 1)
    
    
    
#    action = env.action_space.sample() # your agent here (this takes random actions)
    F = agent.action(t)
    
    
    temp = list(observation)
    temp.append(F)
    observations.append(temp)
#    if F > 0:
#        F = 1
#    else:
#        F = -1
    
    
#    print(F)
    action = np.array([F], dtype='float32')    
    
    try:
        observation, reward, done, info = env.step(action)
    except AssertionError:
        # outside of valid range
        print('assertion error')
        done = True
        observation = env.reset()
        print('time step %s: episode %s: total reward %s time up: %s' % (t, episode, total_r, time_up))
        rewards.append(total_r)
        episode += 1
        time_up = 0
        total_r = 0
        reset = True
#    observation[0] = np.tanh(observation[0])
#    observation[1] = np.tanh(observation[1])
#    observation[3] = np.tanh(observation[3])
#    observation[2] = np.tanh(observation[2])
    agent.X.append(observation) # GARIC observe the environment
    agent.R.append(reward) # GARIC observe the reward
    agent.aen.r_hat(t, done, reset) # GARIC determine internal reinforcement
    agent.aen.backpropagation(t) # GARIC update weights on AEN
    if t > -1:
        agent.asn.backpropagation(agent.aen, agent.sam, t, F)
    reset = False
    t += 1
    time_up += 1
    total_r += reward
    
#    for term in agent.asn.antecedents:
#        print(term)
    
    if done:
        observation = env.reset()
        print('time step %s: episode %s: total reward %s time up: %s' % (t, episode, total_r, time_up))
        rewards.append(total_r)
        episode += 1
        time_up = 0
        total_r = 0
        reset = True
env.close()

#env = gym.make("CartPole-v1")
env = ContinuousCartPoleEnv()
env.min_action = -100
env.max_action = 100
env.seed(20)
env.action_space = gym.spaces.Box(
    low=env.min_action,
    high=env.max_action,
    shape=(1,)
)


aen.X.clear()
aen.R.clear()
aen.R_hat.clear()



observation = env.reset()
agent.X.append(observation) # GARIC observe the environment
agent.R.append(0)
agent.R_hat.append(0)



t = 0
time_up = 0
total_r = 0
episode = 0
reset = True
for idx in range(5):
    NFN_variables[idx].graph(-1,1)
    
print('--- Demo of learned policy ---')
print('continue?')
yes = input()
    
print(observation)
rewards = []
for _ in range(500):
    env.render()
    
    
#    if t > 100 and t % 200 == 0:
#        NFN_variables[4].graph(min(agent.F_primes) - 1, max(agent.F_primes) + 1)
    
    
    
#    action = env.action_space.sample() # your agent here (this takes random actions)
    F = agent.action(t)
    F = agent.Fs[t]
#    if F > 0:
#        F = 1
#    else:
#        F = -1
    
    
#    print(F)
    action = np.array([F], dtype='float32')    
    
    try:
        observation, reward, done, info = env.step(action)
    except AssertionError:
        # outside of valid range
        print('assertion error')
        done = True
        observation = env.reset()
        print('time step %s: episode %s: total reward %s time up: %s' % (t, episode, total_r, time_up))
        rewards.append(total_r)
        episode += 1
        time_up = 0
        total_r = 0
        reset = True
#    observation[0] = np.tanh(observation[0])
#    observation[1] = np.tanh(observation[1])
#    observation[3] = np.tanh(observation[3])
#    observation[2] = np.tanh(observation[2])
    agent.X.append(observation) # GARIC observe the environment
    agent.R.append(reward) # GARIC observe the reward
    agent.aen.r_hat(t, done, reset) # GARIC determine internal reinforcement
#    agent.aen.backpropagation(t) # GARIC update weights on AEN
#    agent.asn.backpropagation(agent.aen, agent.sam, t, F)
    reset = False
    t += 1
    time_up += 1
    total_r += reward
    
#    for term in agent.asn.antecedents:
#        print(term)
    
    if done:
        observation = env.reset()
        print('time step %s: episode %s: total reward %s time up: %s' % (t, episode, total_r, time_up))
        rewards.append(total_r)
        episode += 1
        time_up = 0
        total_r = 0
        reset = True
env.close()