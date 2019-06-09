#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 22:45:27 2019

@author: LuLienHsi
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import numpy as np
import pickle

#Load the saved data
#input1 = open('Policy_SA.pkl','rb')
input1 = open('Policy_QL.pkl','rb')
Policy = pickle.load(input1)

#input2 = open('Qvalue_SA.pkl','rb')
input2 = open('Qvalue_QL.pkl','rb')
Qvalue = pickle.load(input2)

#input3 = open('tmp0_SA.pkl','rb')
input3 = open('tmp0_QL.pkl','rb')
tmp0 = pickle.load(input3)

#input4 = open('tmp1_SA.pkl','rb')
input4 = open('tmp1_QL.pkl','rb')
tmp1 = pickle.load(input4)

#input5 = open('tmp2_SA.pkl','rb')
input5 = open('tmp2_QL.pkl','rb')
tmp2 = pickle.load(input5)

cycle_num = 600
#%%Process data
state_x = []
state_v = []
policy = []
Q_value = [] 
for key, value in Policy.items():
    state_x.append(key[0])
    state_v.append(key[1])
    policy.append(value)   
    Q_value.append(Qvalue[key][value])

#%%Plot Optimal Q value
zero = np.zeros(len(state_x))

fig = plt.figure(figsize=(15,7.5))
#fig.suptitle('2D state sapce & Opimized Q value over the state space - Using SARSA Algorithm')
fig.suptitle('2D state sapce & Opimized Q value over the state space - Using Q-Learning Algorithm')

ax = fig.add_subplot(121, projection='3d')
ax.scatter(state_x,state_v,zero)

ax = fig.add_subplot(122, projection='3d')
ax.scatter(state_x,state_v,Q_value)

ax.set_xlabel('position')
ax.set_ylabel('velocity')
ax.set_zlabel('Q value')

plt.show()

#%%Plot Optimal Policy
state_x0 = []
state_v0 = []
state_x1 = []
state_v1 = []
state_x2 = []
state_v2 = []
for i in range(len(policy)):
    if policy[i] == 0:
        state_x0.append(state_x[i])
        state_v0.append(state_v[i])
    if policy[i] == 1:
        state_x1.append(state_x[i])
        state_v1.append(state_v[i])
    if policy[i] == 2: 
        state_x2.append(state_x[i])
        state_v2.append(state_v[i])

policy0 = np.zeros(len(state_x0))
policy1 = np.ones(len(state_x1))
policy2 = 2*np.ones(len(state_x2))

zero = np.zeros(len(state_x))

fig = plt.figure(figsize=(15,7.5))
#fig.suptitle('2D state sapce & Opimized Policy over the state space - Using SARSA Algorithm')
fig.suptitle('2D state sapce & Opimized Policy over the state space - Using Q-Learning Algorithm')

ax = fig.add_subplot(121, projection='3d')
ax.scatter(state_x,state_v,zero)

ax = fig.add_subplot(122, projection='3d')
ax.scatter(state_x0,state_v0,policy0)
ax.scatter(state_x1,state_v1,policy1)
ax.scatter(state_x2,state_v2,policy2)
ax.legend('012')

ax.set_xlabel('position')
ax.set_ylabel('velocity')
ax.set_zlabel('policy')

plt.show()

#%%
t = np.arange(0,cycle_num)
#(0,0)
fig = plt.figure(figsize=(15,7.5))
#fig.suptitle('Q(x,u) episode varying of (0,0) with action 0,1,2 - Using SARSA Algorithm')
fig.suptitle('Q(x,u) episode varying of (0,0) with action 0,1,2 - Using Q-Learning Algorithm')

ax = fig.add_subplot(131)
action0 = [i[0] for i in tmp0]
ax.plot(t,action0)
ax.set_xlabel('Number of episodes')
ax.set_ylabel('Q value')

ax = fig.add_subplot(132)
action1 = [i[1] for i in tmp0]
ax.plot(t,action1)
ax.set_xlabel('Number of episodes')
ax.set_ylabel('Q value')

ax = fig.add_subplot(133)
action2 = [i[2] for i in tmp0]
ax.plot(t,action2)
ax.set_xlabel('Number of episodes')
ax.set_ylabel('Q value')

plt.show()


#(-1.0,0.05)
fig = plt.figure(figsize=(15,7.5))
#fig.suptitle('Q(x,u) episode varying of (-1.0,0.05) with action 0,1,2 - Using SARSA Algorithm')
fig.suptitle('Q(x,u) episode varying of (-1.0,0.05) with action 0,1,2 - Using Q-Learning Algorithm')

ax = fig.add_subplot(131)
action0 = [i[0] for i in tmp1]
ax.plot(t,action0)
ax.set_xlabel('Number of episodes')
ax.set_ylabel('Q value')

ax = fig.add_subplot(132)
action1 = [i[1] for i in tmp1]
ax.plot(t,action1)
ax.set_xlabel('Number of episodes')
ax.set_ylabel('Q value')

ax = fig.add_subplot(133)
action2 = [i[2] for i in tmp1]
ax.plot(t,action2)
ax.set_xlabel('Number of episodes')
ax.set_ylabel('Q value')
plt.show()

#(0.3,-0.05)
fig = plt.figure(figsize=(15,7.5))
#fig.suptitle('Q(x,u) episode varying of (0.3,-0.05) with action 0,1,2 - Using SARSA Algorithm')
fig.suptitle('Q(x,u) episode varying of (0.3,-0.05) with action 0,1,2 - Using Q-Learning Algorithm')

ax = fig.add_subplot(131)
action0 = [i[0] for i in tmp2]
ax.plot(t,action0)
ax.set_xlabel('Number of episodes')
ax.set_ylabel('Q value')

ax = fig.add_subplot(132)
action1 = [i[1] for i in tmp2]
ax.plot(t,action1)
ax.set_xlabel('Number of episodes')
ax.set_ylabel('Q value')

ax = fig.add_subplot(133)
action2 = [i[2] for i in tmp2]
ax.plot(t,action2)
ax.set_xlabel('Number of episodes')
ax.set_ylabel('Q value')
plt.show()







