#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 16:16:01 2019

@author: LuLienHsi
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import numpy as np
import pickle

#input1 = open('Policy_vi.pkl','rb')
input1 = open('Policy_pi.pkl','rb')
policy = pickle.load(input1)

#input2 = open('tmp0_vi.pkl','rb')
input2 = open('tmp0_pi.pkl','rb')
tmp0 = pickle.load(input2)

#input3 = open('tmp1_vi.pkl','rb')
input3 = open('tmp1_pi.pkl','rb')
tmp1 = pickle.load(input3)

#input4 = open('tmp2_vi.pkl','rb')
input4 = open('tmp2_pi.pkl','rb')
tmp2 = pickle.load(input4)

state_x = []
state_v = []
policy_list= []
Q_value = [] 
for key, value in policy.items():
    state_x.append(key[0])
    state_v.append(key[1])
    policy_list.append(value)   
 
#%%Plot optimal policy
state_x0 = []
state_v0 = []
state_x1 = []
state_v1 = []
state_x2 = []
state_v2 = []
for i in range(len(policy_list)):
    if policy_list[i] == -1:
        state_x0.append(state_x[i])
        state_v0.append(state_v[i])
    if policy_list[i] == 0:
        state_x1.append(state_x[i])
        state_v1.append(state_v[i])
    if policy_list[i] == 1: 
        state_x2.append(state_x[i])
        state_v2.append(state_v[i])

policy0 = -np.ones(len(state_x0))
policy1 = np.zeros(len(state_x1))
policy2 = np.ones(len(state_x2))

zero = np.zeros(len(state_x))

fig = plt.figure(figsize=(15,7.5))
#fig.suptitle('2D state sapce & Opimized Policy (a=-1,b=0,c=1) over the state space - Using Value Iteration Algorithm')
fig.suptitle('2D state sapce & Opimized Policy (a=-1,b=0,c=1) over the state space - Using Policy Iteration Algorithm')

ax = fig.add_subplot(121, projection='3d')
ax.scatter(state_x,state_v,zero)
ax.set_xlabel('angle')
ax.set_ylabel('angular velocity')
ax.set_zlabel('policy')

ax = fig.add_subplot(122, projection='3d')
ax.scatter(state_x0,state_v0,policy0)
ax.scatter(state_x1,state_v1,policy1)
ax.scatter(state_x2,state_v2,policy2)
ax.legend('abc')


ax.set_xlabel('angle')
ax.set_ylabel('angular velocity')
ax.set_zlabel('policy')

plt.show()    


#%%
t = np.arange(0,len(tmp0))
#(0,0)
fig = plt.figure(figsize=(10,5))
fig.suptitle('V(x) of (0,0) over iteration - Using Value Iteration Algorithm')
#fig.suptitle('V(x) of (0,0) over iteration - Using Policy Iteration Algorithm')
plt.plot(t,tmp0)
plt.xlabel('Number of Iteration')
plt.ylabel('V value')
plt.show()

#(0.8727,-1.7453)
fig = plt.figure(figsize=(10,5))
fig.suptitle('V(x) of (0.8727,-1.7453) over iteration - Using Value Iteration Algorithm')
#fig.suptitle('V(x) of (0.8727,-1.7453) over iteration - Using Policy Iteration Algorithm')
plt.plot(t,tmp1)
plt.xlabel('Number of Iteration')
plt.ylabel('V value')
plt.show()


#(4.5379,1.0472)
fig = plt.figure(figsize=(10,5))
fig.suptitle('V(x) of (4.5379,1.0472) over iteration - Using Value Iteration Algorithm')
#fig.suptitle('V(x) of (4.5379,1.0472) over iteration - Using Policy Iteration Algorithm')
plt.plot(t,tmp2)
plt.xlabel('Number of Iteration')
plt.ylabel('V value')
plt.show()

