# -*- coding: utf-8 -*-
"""
Created on Fri May 29 19:10:50 2020

@author: jhost
"""

import gym
import math
import numpy as np
from garic import Term, Variable, Rule, GARIC
from continuous_cartpole import ContinuousCartPoleEnv

# antecedent terms
po1 = Term('PO1', 0.3, 0.3, -1)
ze1 = Term('ZE1', 0.0, 0.3, 0.3)
ne1 = Term('NE1', -0.3, -1, 0.3)
vs1 = Term('VS1', 0.0, 0.05, 0.05)
po2 = Term('PO2', 1.0, 1.0, -1.0)
ze2 = Term('ZE2', 0, 1.0, 1.0)
ne2 = Term('NE2', -1.0, -1.0, 1.0)
vs2 = Term('VS2', 0.0, 0.1, 0.1)
po3 = Term('PO3', 0.5, 0.5, -1.0)
ne3 = Term('NE3', -0.5, -1.0, 0.5)
po4 = Term('PO4', 1.0, 1.0, -1.0)
ne4 = Term('NE4', -1.0, -1.0, 1.0)
ps4 = Term('PS4', 0.0, 0.01, 1.0)
ns4 = Term('NS4', 0.0, 1.0, 0.01)

# consequent terms
pl = Term('PL', 20.0, 5.0, -1.0)
pm = Term('PM', 10.0, 5.0, 6.0)
ps = Term('PS', 5.0, 4.0, 5.0)
pvs = Term('PVS', 1.0, 1.0, 1.0)
nl = Term('NL', -20.0, -1.0, 5.0)
nm = Term('NM', -10.0, 6.0, 5.0)
ns = Term('NS', -5.0, 5.0, 4.0)
nvs = Term('NVS', -1.0, 1.0, 1.0)
ze = Term('ZE', 0.0, 1.0, 1.0)

# antecedent variables
var0 = Variable(0, 'Cart Position', [po3, ne3])
var1 = Variable(1, 'Cart Velocity', [po4, ps4, ne4, ns4])
var2 = Variable(2, 'Pole Angle', [po1, ze1, ne1, vs1])
var3 = Variable(3, 'Pole Velocity At Tip', [po2, ze2, ne2, vs2])

# consequent variables
var4 = Variable(0, 'Force', [pl, pm, ps, pvs, nl, nm, ns, nvs, ze])

inputVariables = [var0, var1, var2, var3]
outputVariables = [var4]

rules = [
        Rule([po1, po2, None, None], [pl]),
        Rule([po1, ze2, None, None], [pm]),
        Rule([po1, ne2, None, None], [ze]),
        Rule([ze1, po2, None, None], [ps]),
        Rule([ze1, ze2, None, None], [ze]),
        Rule([ze1, ne2, None, None], [ns]),
        Rule([ne1, po2, None, None], [ze]),
        Rule([ne1, ze2, None, None], [nm]),
        Rule([ne1, ne2, None, None], [nl]),
        Rule([vs1, vs2, po3, po4], [ps]),
        Rule([vs1, vs2, po3, ps4], [pvs]),
        Rule([vs1, vs2, ne3, ne4], [ns]),
        Rule([vs1, vs2, ne3, ns4], [nvs])
        ]

#rules = [
#        Rule([po1, None, None, None], [pl]),
#        Rule([po1, None, None, None], [pm]),
#        Rule([po1, None, None, None], [ze]),
#        Rule([ze1, None, None, None], [ps]),
#        Rule([ze1, None, None, None], [ze]),
#        Rule([ze1, None, None, None], [ns]),
#        Rule([ne1, None, None, None], [ze]),
#        Rule([ne1, None, None, None], [nm]),
#        Rule([ne1, None, None, None], [nl]),
#        Rule([vs1, None, po3, None], [ps]),
#        Rule([vs1, None, po3, None], [pvs]),
#        Rule([vs1, None, ne3, None], [ns]),
#        Rule([vs1, None, ne3, None], [nvs])
#        ]

agent = GARIC(inputVariables, outputVariables, rules, 5)

#env = gym.make("CartPole-v1")
env = ContinuousCartPoleEnv()
env.min_action = -100
env.max_action = 100
env.action_space = gym.spaces.Box(
    low=env.min_action,
    high=env.max_action,
    shape=(1,)
)

observation = env.reset()
agent.X.append(observation) # GARIC observe the environment
agent.R.append(0)
agent.R_hat.append(0)
t = 0
time_up = 0
total_r = 0
episode = 0
reset = True
rewards = [] # a list where the ith element is the total reward obtained during the ith episode
print(observation)
#for inputVariable in inputVariables:
#    inputVariable.graph(-1.5,1.5)
var0.graph(-0.51,0.51)
var1.graph(-1.01,1.01)
var2.graph(-0.4,0.4)
var3.graph(-1.01,1.01)
var4.graph()
for _ in range(1000):
    env.render()
#    action = env.action_space.sample() # your agent here (this takes random actions)
    F = np.array([agent.action(t)], dtype='float32')
#    print('apply %s force' % F)
    observation, reward, done, info = env.step(F)
    observation[0] /= 4.8
    observation[1] = np.tanh(observation[1])
    observation[3] = np.tanh(observation[3])
    observation[2] /= 140
    agent.X.append(observation) # GARIC observe the environment
    agent.R.append(reward) # GARIC observe the reward
    agent.aen.r_hat(t, done, reset) # GARIC determine internal reinforcement
    agent.aen.backpropagation(t) # GARIC update weights on AEN
    agent.asn.backpropagation(agent.aen, agent.sam, t)
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