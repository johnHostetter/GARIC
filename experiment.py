# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 11:32:18 2020

@author: jhost
"""

import random
from testbed import randomPlay
random.seed(0)
# seed = 11 is very great
data = randomPlay(seed=11)
from empiricalfuzzyset import EmpiricalFuzzySet
features = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Velocity At Tip', 'Force']
efs = EmpiricalFuzzySet(features)
NFN_variables = efs.main(data)
from ruleGeneration import init_rules
rules = init_rules(NFN_variables, data.episodes)
from testbed import demo
# for random play seed = 11, seed = 6, 7, 8, 9 (original) or 10
data_1, agent = demo(data.aen, NFN_variables, rules, data, explore=False, seed=9)
#from testbed import playback
#playback(X)