# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 11:32:18 2020

@author: jhost
"""

import random
from testbed import randomPlay
random.seed(0)
# seed = 11 is great
data = randomPlay(seed=11)
from empiricalfuzzyset import EmpiricalFuzzySet
features = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Velocity At Tip', 'Force']
efs = EmpiricalFuzzySet(features)
NFN_variables = efs.main(data)
from ruleGeneration import init_rules
rules = init_rules(NFN_variables, data.X, data.V)
from testbed import demo
# for random play seed = 11, seed = 6, 7, 8, 9 (original) or 10
X = demo(data.aen, NFN_variables, rules, data.X, data.V, explore=False, seed=7)
#from testbed import playback
#playback(X)