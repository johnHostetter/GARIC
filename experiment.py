# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 11:32:18 2020

@author: jhost
"""

# NOTE TO SELF: before executing this script, you must delete the local __pycache__/ folder
import random
import numpy as np
from testbed import randomPlay
np.random.seed(0)
random.seed(0)
# seed = 11 is very great
data = randomPlay(seed=11, angle=32)
from empiricalfuzzyset import EmpiricalFuzzySet
features = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Velocity At Tip', 'Force']
efs = EmpiricalFuzzySet(features)
NFN_variables = efs.main(data)
from ruleGeneration import init_rules
rules, consequent_terms = init_rules(NFN_variables, data.episodes)
NFN_variables[4].terms = consequent_terms
from testbed import demo
# for random play seed = 11, seed = 6, 7, 8, 9 (original) or 10
data_1, agent = demo(data.aen, NFN_variables, rules, data, time=6000, explore=False, seed=9, angle=32)
#from testbed import playback
#playback(X)