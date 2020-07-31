# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:52:22 2020

@author: jhost
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from collections import Counter
from statistics import stdev 
from neurofuzzynetwork import Term, Variable, NFN_bellShapedMembership

np.random.seed(0)

class EmpiricalFuzzySet():
    def __init__(self, features):
        self.features = features
        self.numerator = None
        self.distances = np.array([])
        self.powers = np.array([])
        self.new_p_idxs = -1
        
    def unimodalDensity(self, X, i, dist):
        """ Calculate the unimodal density of a particular ith data sample
        from the set of observations, X, with the distance metric, dist. """
        K = len(X)
        denominator = 0
        
        if len(self.distances) == 0:
            print('Building matrices...')
            self.distances = np.array([float('inf')]*len(X)*len(X)).reshape(len(X), len(X))
            self.powers = np.array([float('inf')]*len(X)*len(X)).reshape(len(X), len(X))
            print('Done.')
            print('Standby for initialization...')
        
        if self.numerator == None:
            self.numerator = 0
            for k in range(K):
                for j in range(K):
                    self.numerator += pow(dist(X[k], X[j]), 2)
        for j in range(K):
            if self.distances[i, j] != float('inf') or self.distances[j, i] != float('inf'):
                denominator += self.distances[i, j]
            else:    
                self.distances[i, j] = dist(X[i], X[j])
                self.powers[i, j] = pow(self.distances[i, j], 2)
                self.distances[j, i] = self.distances[i, j]
                self.powers[j, i] = self.powers[i, j]
                denominator += self.powers[i, j]
            
        denominator *= 2 * K
        return self.numerator / denominator
    
    def unimodalDensity1(self, X, x, dist):
        """ Calculate the unimodal density of a particular ith data sample
        from the set of observations, X, with the distance metric, dist. """
        K = len(X)
        denominator = 0
        
        if len(self.distances) == 0:
            print('Building matrices...')
            self.distances = np.array([float('inf')]*len(X)*len(X)).reshape(len(X), len(X))
            self.powers = np.array([float('inf')]*len(X)*len(X)).reshape(len(X), len(X))
            print('Done.')
            print('Standby for initialization...')
        
        if self.numerator == None:
            self.numerator = 0
            for k in range(K):
                for j in range(K):
                    self.numerator += pow(dist(X[k], X[j]), 2)
        for j in range(K):
            denominator += pow(dist(x, X[j]), 2)
            
        denominator *= 2 * K
        return self.numerator / denominator

    def multimodalDensity(self, X, U, F, i, dist):
        """ Calculate the multimodal density of a particular ith data sample
        from the set of observations, X, with the distance metric, dist, and
        using the set of frequencies, F. """
    #    idx = X.index(U[i])
        idx = i
        return F[i] * self.unimodalDensity(X, idx, dist)
    
    def multimodalDensity1(self, X, x, dist):
        """ Calculate the multimodal density of a particular ith data sample
        from the set of observations, X, with the distance metric, dist, and
        using the set of frequencies, F. """
    #    idx = X.index(U[i])
        return self.unimodalDensity1(X, x, dist)

    def unique(self, X):
        counter = Counter(X)
        U = list(counter.keys()) # unique observations
        F = list(counter.values()) # frequencies
        return (U, F)
    
    def plotDistribution(self, X, Y, title):
        plt.ticklabel_format(useOffset=False, style='plain')
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(X, Y, 'o', color='blue')
        plt.legend()
        
    def objectiveMethod(self, data):
        #X.sort()
        #U, F = unique(X)
        U = data.X
        F = np.array([1]*len(data.X))
        
        x_lst = []
        y_lst = []
        densities = {}
        
        # step 1
        
        for i in range(len(U)):
            mm = self.multimodalDensity(data.X, U, F, i, distance.euclidean)
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
        uR = u1_star # assigned to but never used
        
        ctr = 0
        visited = {}
        previousLeftIndex = 0
        previousLeftValue = 0
        localMaximaIndexes = []
        prototypes = []
        while len(visited.keys()) < len(U):
            # step 4
            srted = sorted(self.distances[idx])
            for uRidx in range(len(srted)):
                item = srted[uRidx] # assigned to but never used
                if uRidx == idx:
                    continue
                if uRidx in visited.keys():
                    continue
                else:
                    visited[uRidx] = ctr
                    ctr += 1
                    ULstar.append(U[uRidx])
                    idx = uRidx # step 5, now go back to step 4
                    mm = self.multimodalDensity(data.X, U, F, uRidx, distance.euclidean)
                    DMM.append(mm) # step 6
                    # step 7 is both if & else statements
                    if mm > previousLeftValue:
                        previousLeftIndex = uRidx
                        previousLeftValue = mm
                    else:
                        localMaximaIndexes.append(uRidx - 1)
                        prototypes.append(U[uRidx - 1])
                        previousLeftIndex = uRidx # assigned to but never used
                        previousLeftValue = mm
                    break
        #    row = idx
        #    col = 0
        #    U = np.delete(U, row, col)
        
        # step 8
        clouds = {} # each element is a list that is indexed by a prototype index
        labels = [] # a direct labeling where the ith element of X has an ith label
        for x in data.X:
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
            dmm_p0[prototype_idx] = self.multimodalDensity1(data.X, p0[prototype_idx], distance.euclidean)
            
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
                    if self.multimodalDensity1(data.X, q, distance.euclidean) > max_val:
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
        for x in data.X:
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
                        
        # additional steps required
        variables = self.make_variables(data, p0, clouds)
        return variables, clouds
    
    def main(self, data):
        variables, clouds = self.objectiveMethod(data)
        self.compress_variables(clouds, variables)
        NFN_variables = self.make_NFN_variables(variables)
        return NFN_variables
        
    def gaussianMembership(self, x, center, sigma):
        numerator = (-1) * pow(x - center, 2)
        denominator = 2 * pow(sigma, 2)
        return pow(math.e, numerator / denominator)
    
    def mystdev(self, lst, i):
        """ Calculate the standard deviation of the ith feature
        from a list of observations that have been collected. """
        new_lst = []
        for item in lst:
            new_lst.append(item[i])
        return stdev(new_lst)
    
    def distMatrix(self, terms):
        matrix = np.array([float('inf')]*len(terms)*len(terms)).reshape(len(terms), len(terms))
        for i in terms.keys():
            for j in terms.keys():
                i_matrix = np.where(np.array(list(terms.keys()))==i)[0][0]
                j_matrix = np.where(np.array(list(terms.keys()))==j)[0][0]
                matrix[i_matrix, j_matrix] = distance.euclidean(terms[i]['center'], terms[j]['center'])
        return matrix

    def identifySimilarPair(self, terms, distMatrix):
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

    def reduction(self, clouds, terms, similarity, feature_idx):
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
            clouds[self.new_p_idxs] = clouds_new
            sig = self.mystdev(clouds_new, feature_idx)
        #    clouds.pop(terms[A]['p_idx'])
        #    clouds.pop(terms[B]['p_idx'])
            del terms[A]
            del terms[B]
            term_new = {'center':c_new, 'sigma':sig, 'support':support_C, 'p_idx':self.new_p_idxs}
            terms[self.new_p_idxs] = term_new
            self.new_p_idxs -= 1
            return term_new
        return None

    def compression(self, clouds, threshold, num_of_terms, terms, feature_idx):
        matrix = self.distMatrix(terms[feature_idx])
        similarity = self.identifySimilarPair(terms[feature_idx], matrix)
#        print('num of terms remaining: %s' % len(terms[feature_idx]))
        if len(terms[feature_idx]) > num_of_terms or similarity['distance'] < threshold:
            result = self.reduction(clouds, terms[feature_idx], similarity, feature_idx)
            if result == None:
                return False # stop, do not continue
            else:
                return True # continue
        return False # stop, do not continue

    def make_variables(self, data, p0, clouds):
        variables = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}}
        for feature_idx in range(len(self.features)):
            x_lst = []
            mu_lst = []
            for p_idx in p0.keys():
                if len(clouds[p_idx]) > 1:
                    c = p0[p_idx][feature_idx]
                    sig = self.mystdev(clouds[p_idx], feature_idx)
                    variables[feature_idx][p_idx] = {'center':c, 'sigma':sig, 'support':len(clouds[p_idx]), 'p_idx':p_idx}
                    for x in data.X:
                        x = x[feature_idx]
                        mu = self.gaussianMembership(x, c, sig)
                        x_lst.append(x)
                        mu_lst.append(mu)
        return variables

    def compress_variables(self, clouds, variables):
        print('--- COMPRESSING LINGUISTIC TERMS ---')
        thresholds = [0.01, 0.01, 0.001, 0.01, 0.01] # the corresponding threshold for each feature
        for feature_idx in range(len(self.features)):
            cont = True
            while cont:
                if feature_idx != 4:
                    cont = self.compression(clouds, thresholds[feature_idx], 7, variables, feature_idx) # 7 has performed well, achieved 125 max reward
                else:
                    cont = self.compression(clouds, thresholds[feature_idx], 7, variables, feature_idx) # 7 has performed well, achieved 125 max reward

    def plot_variables(self, data, variables):
        for feature_idx in range(len(self.features)):
            x_lst = []
            mu_lst = []
            for p_idx in variables[feature_idx].keys():
                c = variables[feature_idx][p_idx]['center']
                sig = variables[feature_idx][p_idx]['sigma']
                for x in data.X:
                    x = x[feature_idx]
                    mu = self.gaussianMembership(x, c, sig)
                    x_lst.append(x)
                    mu_lst.append(mu)
#                title = self.features[feature_idx]
#                self.plotDistribution(x_lst, mu_lst, title)
#            plt.show()

    # trying to incorporate empirical fuzzy sets into neuro fuzzy networks
    
    def make_NFN_variables(self, variables):
        NFN_variables = [] # variables meant for the neuro fuzzy network
        term_labels = ['Very Negative', 'Negative', 'Slightly Negative', 'Moderate', 'Slightly Positive', 'Positive', 'Very Positive']
        all_terms = []
        for var_key in variables.keys():
            idx = var_key
            var_label = self.features[var_key]
            terms = []
            term_label_idx = 0
        
            for term_key in variables[var_key].keys():
                c = variables[var_key][term_key]['center']
                sig = variables[var_key][term_key]['sigma']
                sup = variables[var_key][term_key]['support']
                params = {'center':c, 'sigma':sig}
                term_label = term_labels[term_label_idx]
                
                term = Term(var_key, NFN_bellShapedMembership, params, sup, term_label, var_label)
                
                term_label_idx += 1
                terms.append(term)
            all_terms.append(terms)
            variable = Variable(idx, terms, var_label)
            NFN_variables.append(variable)
        return NFN_variables