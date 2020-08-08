# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 19:07:39 2020

@author: jhost
"""

import gym
import copy
import numpy as np
from garic import AEN
from ruleGeneration import update_rules
from neurofuzzynetwork import eFL_ACC
from continuous_cartpole import ContinuousCartPoleEnv

#np.random.seed(0)

class Data():
    def __init__(self, X, aen, episodes):
        self.X = X
        self.aen = aen
        self.episodes = episodes        
        
def normalization(observation):
    for i in range(len(observation)): 
        observation[i] = np.tanh(observation[i])
    return observation

def addActionToObservation(observation, action):
    """ Append the action to the observation. """
    observation = list(observation)
    observation.append(action[0])
    observation = np.array(observation)
    return observation

def initAEN(observation):
    aen = AEN([], [], [], 4, h=5)
    aen.X.append(observation) # GARIC observe the environment
    aen.R.append(0)
    aen.R_hat.append(0)
    return aen

def updateAEN(aen, t, observation, reward, done, reset):
    """ Update AEN only for demo random play. """
    aen.X.append(observation) # GARIC observe the environment
    aen.R.append(reward) # GARIC observe the reward
    aen.r_hat(t, done, reset) # GARIC determine internal reinforcement
    aen.backpropagation(t) # GARIC update weights on AEN
    return aen

def loadAEN(agent, aen):
    agent.aen = aen # testing loading up a pre-trained AEN
    aen.X = agent.X
    aen.R = agent.R
    aen.R_hat = agent.R_hat
    return agent

def initAgent(agent, observation):
    agent.X.append(observation) # GARIC observe the environment
    agent.R.append(0)
    agent.R_hat.append(0)
    return agent

def updateAgent(agent, t, observation, reward, done, reset, F, backpropagate):
    agent.X.append(observation) # GARIC observe the environment
    agent.R.append(reward) # GARIC observe the reward
    agent.aen.r_hat(t, done, reset) # GARIC determine internal reinforcement
    agent.aen.backpropagation(t) # GARIC update weights on AEN
    if backpropagate:
        agent.asn.backpropagation(agent.aen, agent.sam, t, F)
    return agent
        
def initEnv(seed, angle, min_action, max_action):
    env = ContinuousCartPoleEnv(seed, angle)
    env.seed = seed
    env.min_action = min_action
    env.max_action = max_action
    env.action_space = gym.spaces.Box(
        low=env.min_action,
        high=env.max_action,
        shape=(1,)
    )
    env.action_space.seed(seed)
    return env

def graph(NFN_variables):
    for idx in range(len(NFN_variables)):
        NFN_variables[idx].graph(-1,1)

def prompt():
    print('--- Demo of exploration and fine tuning of membership functions ---')
    print('continue?')
    input()
    
def randomPlay(seed, angle):
    """ Generate random gameplay for initialization of learning method. """
    X = []
    steps = []
    env = initEnv(seed, angle, -1, 1)
    observation = env.reset()
    reset = True
    
    episodes = []
    time_to_episodes = {}
    total_reward = 0.0 # total reward for this episode so far
    
    aen = initAEN(observation)
    
    rewards = []
    for t in range(200):
        env.render()
        action = env.action_space.sample() # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        observation = normalization(observation)
        observation = addActionToObservation(observation, action)
        aen = updateAEN(aen, t, observation, reward, done, reset)
        
        # add to episode history
        total_reward += reward
        X.append(observation)
        steps.append(observation)
        time_to_episodes[t] = len(episodes)

        reset = False
        if done:
            observation = env.reset()
            episodes.append(steps)
            rewards.append(total_reward)
            total_reward = 0.0
            steps = []
            reset = True
            
    if not(done):
        observation = env.reset()
        episodes.append(steps)
        total_reward = 0.0
        steps = []
        reset = True
    
    env.close()
    return Data(X, aen, episodes)

def demo(aen, NFN_variables, rules, data, time, explore, seed, angle):
    """ Demo of learning method. """
    X = []
    steps = []
    agent = eFL_ACC(NFN_variables[:4], NFN_variables[4:], rules, 5, lower=-2, upper=2)    
    env = initEnv(seed, angle, -100, 100)
    observation = env.reset()
    reset = True
    done = False
    
    agent = loadAEN(agent, aen)
    agent = initAgent(agent, observation)
    
    episodes = []
    time_to_episodes = {}
    total_reward = 0.0 # total reward for this episode so far

        
    graph(NFN_variables)
    prompt()
            
    rewards = []
    backpropagate = True
    for t in range(time):
        env.render()
#        if done and len(episodes) % 5 == 0 and not(backpropagate):
#            print('Updating fuzzy logic control rules...')
#            total_episodes = copy.deepcopy(episodes)
#            total_episodes.extend(data.episodes)
#            rules = update_rules(NFN_variables, total_episodes, rules)
#            agent.asn.updateRules(rules)
    
        F = agent.action(t, explore)
        action = np.array([F], dtype='float32')    
        observation, reward, done, info = env.step(action)

        # full tanh normalization resulted in mean of 23.65 and max of 76 with rule update cutoff at score of 60
        observation = normalization(observation)
        
        # add to episode history
        total_reward += reward
        X.append(observation)
        steps.append(observation)
        time_to_episodes[t] = len(episodes)
        
#        observation = addActionToObservation(observation, action)
#        if t == 3000:
#            backpropagate = True
        agent = updateAgent(agent, t, observation, reward, done, reset, F, backpropagate)
        
        reset = False
        if done:
            observation = env.reset()
            print('time step %s: episode %s: total reward %s' % (t, len(episodes), total_reward))
            episodes.append(steps)
            rewards.append(total_reward)
            total_reward = 0.0
            steps = []
            reset = True

#        if done:
##            avg_reward = total_reward / len(steps)
#            episodes[episode_ctr] = {'steps':steps, 'value':total_reward}
#            observation = env.reset()
#            total_reward = 0.0
#            episode_ctr += 1
#            steps = []
#            reset = True
#            
#    if not(done):
##        avg_reward = total_reward / len(steps)
#        episodes[episode_ctr] = {'steps':steps, 'value':total_reward}
#        observation = env.reset()
#        total_reward = 0.0
#        episode_ctr += 1
#        steps = []

    env.close()
    print('Max reward: %s' % max(rewards))
    print('Mean reward: %s' % np.mean(rewards))
    return Data(agent.X, agent.aen, episodes), agent
    
def playback(X):
    print('Playback of events...')
    env = ContinuousCartPoleEnv()
    for x in X:
        env.state = x[:4]
        env.render()
    env.close()