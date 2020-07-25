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

np.random.seed(0)

class Data():
    def __init__(self, X, V, aen):
        self.X = X
        self.V = V
        self.aen = aen

def randomPlay(seed):
    """ Generate random gameplay for initialization of learning method. """
    X = []
    V = [] # reward thus far
    
    env = ContinuousCartPoleEnv(seed)
    env.seed = seed
    env.min_action = -1.0
    env.max_action = 1.0
    env.action_space = gym.spaces.Box(
        low=env.min_action,
        high=env.max_action,
        shape=(1,)
    )
    env.action_space.seed(seed)
    
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
    
    for t in range(200):
        env.render()
        action = env.action_space.sample() # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        observation[0] = np.tanh(observation[0])
        observation[1] = np.tanh(observation[1])
        observation[3] = np.tanh(observation[3])
        observation[2] = np.tanh(observation[2])
        
        # add force to observation
        observation = list(observation)
        observation.append(action[0])
        observation = np.array(observation)
        X.append(np.array(observation))
        V.append(total_reward)
        
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
    return Data(X, V, aen)

def demo(aen, NFN_variables, rules, init_X, init_V, explore, seed):
    """ Demo of learning method. """
    agent = eFL_ACC(NFN_variables[:4], NFN_variables[4:], rules, 5, lower=-2, upper=2)    

    env = ContinuousCartPoleEnv(seed)
    env.seed = seed
    env.min_action = -100.0
    env.max_action = 100.0
    env.action_space = gym.spaces.Box(
        low=env.min_action,
        high=env.max_action,
        shape=(1,)
    )
    env.action_space.seed(seed)
    
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
    t_to_episode = {}
    episode_vals = {}
    reset = True
    for idx in range(5):
        NFN_variables[idx].graph(-1,1)
        
    new_V = []
    
    print('--- Demo of exploration and fine tuning of membership functions ---')
    print('continue?')
    input()
    
    observations = []
        
    print(observation)
    rewards = []
    for _ in range(2000):
        env.render()
        if episode % 20 == 0 and reset and t <= 2000 and len(rewards) > 0 and max(rewards) < 80: # every 10 to 20 episodes
#        if episode % 10 == 0 and reset and t <= 2000 and len(rewards) > 0 and max(rewards) < 90: # every 10 to 20 episodes
            print('Updating fuzzy logic control rules...')
            
            
            test_V = []
            
            for i in range(len(t_to_episode)):
                test_V.append(episode_vals[t_to_episode[i]])
            
            total_X = copy.deepcopy(init_X)
            total_X.extend(observations)
            total_V = copy.deepcopy(init_V)
            total_V.extend(test_V)
            rules = update_rules(NFN_variables, total_X[-3000:], total_V[-3000:])
            agent.asn.updateRules(rules)
    
        F = agent.action(t, explore)
        
        temp = list(observation)
        temp.append(F)
        observations.append(temp)
        
        t_to_episode[t] = episode
    
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
            episode_vals[episode] = total_r
            episode += 1
            time_up = 0
            total_r = 0
            reset = True
        # full tanh normalization resulted in mean of 23.65 and max of 76 with rule update cutoff at score of 60
        observation[0] = np.tanh(observation[0])
        observation[1] = np.tanh(observation[1])
        observation[3] = np.tanh(observation[3])
        observation[2] = np.tanh(observation[2])
        agent.X.append(observation) # GARIC observe the environment
        agent.R.append(reward) # GARIC observe the reward
        agent.aen.r_hat(t, done, reset) # GARIC determine internal reinforcement
        agent.aen.backpropagation(t) # GARIC update weights on AEN
        if len(rewards) > 0 and max(rewards) >= 60:
            agent.asn.backpropagation(agent.aen, agent.sam, t, F)
        reset = False
        t += 1
        time_up += 1
        total_r += reward
        
        new_V.append(total_r)
        
        if done:
            observation = env.reset()
            print('time step %s: episode %s: total reward %s time up: %s' % (t, episode, total_r, time_up))
            rewards.append(total_r)
            episode_vals[episode] = total_r
            episode += 1
            time_up = 0
            total_r = 0
            reset = True
    env.close()
    print('Max reward: %s' % max(rewards))
    print('Mean reward: %s' % np.mean(rewards))
    return agent.X
    
def playback(X):
    print('Playback of events...')
    env = ContinuousCartPoleEnv()
    for x in X:
        env.state = x[:4]
        env.render()
    env.close()