#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 22:26:22 2019

@author: LuLienHsi
"""

import numpy as np 
from random import randint 
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

class Optimizer:
    def __init__(self):
        self.states = np.arange(25) + 1
        self.gamma = 0.99

        #key: states , value: [north, south, east, west] new state x'
        self.control_result = {1:[1,6,2,1], 2:[22,22,22,22], 3:[3,8,4,2], 4:[14,14,14,14], 5:[5,10,5,4],
                               6:[1,11,7,6], 7:[2,12,8,6], 8:[3,13,9,7], 9:[4,14,10,8], 10:[5,15,10,9],
                               11:[6,16,12,11], 12:[7,17,13,11], 13:[8,13,14,12], 14:[9,19,15,13], 15:[10,20,15,14],
                               16:[11,21,17,16], 17:[12,22,18,16], 18:[13,23,19,17], 19:[14,24,20,18], 20:[15,25,20,19],
                               21:[16,21,22,21], 22:[17,22,23,21], 23:[18,23,24,22], 24:[19,24,25,23], 25:[20,25,25,24]}
 
        #key: states , value: [north, south, east, west] stage cost 
        self.stage_cost = {1:[1,0,0,1], 2:[-10,-10,-10,-10], 3:[1,0,0,0], 4:[-5,-5,-5,-5], 5:[1,0,1,0],
                           6:[0,0,0,1], 7:[0,0,0,0], 8:[0,0,0,0], 9:[0,0,0,0], 10:[0,0,1,0],
                           11:[0,0,0,1], 12:[0,0,0,0], 13:[0,0,0,0], 14:[0,0,0,0], 15:[0,0,1,0],
                           16:[0,0,0,1], 17:[0,0,0,0], 18:[0,0,0,0], 19:[0,0,0,0], 20:[0,0,1,0],
                           21:[0,1,0,1], 22:[0,1,0,0], 23:[0,1,0,0], 24:[0,1,0,0], 25:[0,1,1,0]} 
        self.control_num = 4
        self.state_num = 25
        
    def value_iteration(self):
        #Initialize the random guess value for all states 
        v = np.zeros(self.state_num)
        policy = np.zeros(self.state_num)
        count = 0
        value_iteration = 0
        while count != self.state_num:
            count = 0 
            for i in self.states:
                index = i - 1
                v_temp = v[index]
                #Policy Improvement
                #compute four controls value, 1 is the probability of the stochastic motion model 
                north = self.stage_cost[i][0] + self.gamma * 1 * v[self.control_result[i][0]-1]
                south = self.stage_cost[i][1] + self.gamma * 1 * v[self.control_result[i][1]-1]
                east = self.stage_cost[i][2] + self.gamma * 1 * v[self.control_result[i][2]-1]
                west = self.stage_cost[i][3] + self.gamma * 1 * v[self.control_result[i][3]-1]
                policy[index] = int(np.argmin([north,south,east,west]))
                p = int(policy[index])
                
                #Value Update
                v[index] = self.stage_cost[i][p] + self.gamma * 1 * v[self.control_result[i][p]-1]
                
                if v[index] == v_temp: 
                    count += 1
            value_iteration += 1   
        policy += 1 
            
        policy_vi = policy
        policy_vi = [int(i) for i in policy_vi]
        value_vi = v
        return policy_vi, value_vi, value_iteration

    def policy_iteration(self):
        #Guess some initial policy for every states and iterate on the policy until the policy is optimal 
        policy = np.zeros(self.state_num)
        for i in range(self.state_num):
            policy[i]= randint(0, self.control_num - 1)
    
        v = np.zeros((self.state_num ,1))  
        l = np.zeros((self.state_num ,1))  
        count = 0
        policy_iteration = 0
        v_temp = v

        while count != self.state_num: 
            #Policy Evaluation --> solve the linear equation
            #Get the transition probability matrix
            P = np.zeros((self.state_num ,self.state_num ))
            for i in self.states: 
                index = i - 1
                next_state = self.control_result[i][int(policy[index])]
                P[index][next_state - 1] = 1
                l[index] = self.stage_cost[i][int(policy[index])] 
                
            inverse_term = np.linalg.inv(np.identity(self.state_num ) - self.gamma * P)
            v = np.matmul(inverse_term, l)    
            
            #Policy Improvement 
            for i in self.states:  
                index = i - 1
                north = self.stage_cost[i][0] + self.gamma * 1 * v[self.control_result[i][0] -1]
                south = self.stage_cost[i][1] + self.gamma * 1 * v[self.control_result[i][1] -1]
                east = self.stage_cost[i][2] + self.gamma * 1 * v[self.control_result[i][2] -1]
                west = self.stage_cost[i][3] + self.gamma * 1 * v[self.control_result[i][3] -1]
                policy[index] = np.argmin([north,south,east,west])
                
            count = 0 
            for i in range(self.state_num):
                if v_temp[i] == v[i]:
                    count+= 1
                v_temp[i] = v[i]
            policy_iteration += 1 
        policy += 1
        policy_pi = policy   
        policy_pi = [int(i) for i in policy_pi]
        value_pi = np.transpose(v)[0]
        
        return policy_pi, value_pi, policy_iteration
    
    def Qvalue_iteration(self):
        Q = np.zeros((self.state_num ,self.control_num ))
        Qvalue_iteration = 0
        count = 0 
        #Update Q value of the four controls for each state
        while count != self.state_num * self.control_num :
            count = 0       
            for i in self.states:
                index = i - 1
                for q in range(self.control_num):
                    Q_temp = Q[index][q]
                    next_state = self.control_result[i][q]
                    Q[index][q] = self.stage_cost[i][q] + self.gamma * 1 * Q[next_state-1][np.argmin(Q[next_state-1])]
                
                    if Q[index][q] == Q_temp: 
                        count += 1
            Qvalue_iteration += 1   
            
        #Extract optimal policy and opitmal value
        policy_qvi = []
        value_qvi = []
        for i in self.states:
            index = i - 1
            optimal_policy =int(np.argmin(Q[index]) + 1)
            optimal_value = np.min(Q[index])
            policy_qvi.append(optimal_policy)
            value_qvi.append(optimal_value)
        
        return policy_qvi, value_qvi, Qvalue_iteration

if __name__ == '__main__':
    optimizer = Optimizer()
    policy_vi, value_vi, iternum_vi = optimizer.value_iteration()
    policy_pi, value_pi, iternum_pi = optimizer.policy_iteration()
    policy_qvi, value_qvi, iternum_qvi = optimizer.Qvalue_iteration()
    
    #Plot 
    fig = plt.figure(figsize=(14,7))
    fig.suptitle('Optimal Value & Policy of Value Iteration (gamma = 0.99)')
    value_vi = np.reshape(value_vi,(5,5))
    policy_vi = np.reshape(policy_vi,(5,5))
    
    with sns.axes_style("white"):
        ax = fig.add_subplot(1,2,1)
        ax = sns.heatmap(value_vi, cmap='gray',annot=True,fmt='.3f')
        
        ax = fig.add_subplot(1,2,2)
        ax = sns.heatmap(policy_vi, cmap='gray',annot=True,fmt='d',cbar=False)
        plt.show()
        
    
    fig = plt.figure(figsize=(14,7))
    fig.suptitle('Optimal Value & Policy of Policy Iteration (gamma = 0.99)')
    value_pi = np.reshape(value_pi,(5,5))
    policy_pi = np.reshape(policy_pi,(5,5))
    
    with sns.axes_style("white"):
        ax = fig.add_subplot(1,2,1)
        ax = sns.heatmap(value_pi, cmap='gray',annot=True,fmt='.3f')
        
        ax = fig.add_subplot(1,2,2)
        ax = sns.heatmap(policy_pi, cmap='gray',annot=True,fmt='d',cbar=False)
        plt.show()
        
    fig = plt.figure(figsize=(14,7))
    fig.suptitle('Optimal Value & Policy of Q-value Iteration (gamma = 0.99)')
    value_qvi = np.reshape(value_qvi,(5,5))
    policy_qvi = np.reshape(policy_qvi,(5,5))
    
    with sns.axes_style("white"):
        ax = fig.add_subplot(1,2,1)
        ax = sns.heatmap(value_qvi, cmap='gray',annot=True,fmt='.3f')
        
        ax = fig.add_subplot(1,2,2)
        ax = sns.heatmap(policy_qvi, cmap='gray',annot=True,fmt='d',cbar=False)
        plt.show()

 



    
    
    