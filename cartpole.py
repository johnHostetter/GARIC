# -*- coding: utf-8 -*-
"""
Created on Fri May 29 19:10:50 2020

@author: jhost
"""

import gym
env = gym.make("CartPole-v1")
observation = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    
    if done:
        observation = env.reset()
env.close()