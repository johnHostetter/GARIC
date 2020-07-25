# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 11:32:18 2020

@author: jhost
"""

from testbed import randomPlay
data = randomPlay()
from empiricalfuzzyset import EmpiricalFuzzySet
features = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Velocity At Tip', 'Force']
efs = EmpiricalFuzzySet(features)
NFN_variables = efs.main(data)
from ruleGeneration import init_rules
rules = init_rules(NFN_variables, data.X, data.V)
from testbed import demo
X = demo(data.aen, NFN_variables, rules, data.X, data.V, explore=False)
#from testbed import playback
#playback(X)