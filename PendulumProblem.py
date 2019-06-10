"""
==================================
Inverted pendulum animation class
==================================

Collaborat with Hung-Ting Chen 

Adapted from the double pendulum problem animation.
https://matplotlib.org/examples/animation/double_pendulum_animated.html
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import multivariate_normal
from scipy.stats import norm
import random
from random import randint 
import pickle


class EnvAnimate:

    '''
    Initialize Inverted Pendulum
    '''
    def __init__(self):

        # Change this to match your discretization
        self.dt = 0.5
        self.time = 60.0
        self.t = np.arange(0.0, self.time, self.dt)
        
        
        #Discretize the space
        self.angle_max = 2*np.pi
        self.angle_min = 0
        self.angle_resolution = 2*np.pi*(10/360) #10 degree for a state
        self.n1 = int((self.angle_max - self.angle_min) / self.angle_resolution) #number of states of angle
        
        self.n2 = 10 #number of states of angle velocity 
        self.angle_vel_max = (2*np.pi/self.n1)*(self.n2/self.dt)/2
        self.angle_vel_min = -self.angle_vel_max
        self.angle_vel_resolution = (self.angle_vel_max - self.angle_vel_min) / self.n2
        
        self.round = 4
        
        #Start discretizing
        #Initialize the V value
        #Initialize the policy 
        self.states = {} 
        self.value = {} 
        self.policy = {}
        for i in range(self.n1):
            for j in range(self.n2): 
                self.states[(round(self.angle_min + i*self.angle_resolution,self.round), round(self.angle_vel_min + j*self.angle_vel_resolution,self.round))] \
                    = [[round(self.angle_min + i*self.angle_resolution,self.round)], [round(self.angle_vel_min + j*self.angle_vel_resolution,self.round)]]
                self.value[(round(self.angle_min + i*self.angle_resolution,self.round), round(self.angle_vel_min + j*self.angle_vel_resolution,self.round))] \
                    = np.random.uniform(low = -1, high = 1)
                self.policy[(round(self.angle_min + i*self.angle_resolution,self.round), round(self.angle_vel_min + j*self.angle_vel_resolution,self.round))] \
                    = 0 
                    
        self.state_num = len(self.states)
        
        #Control space
        #Use env.action_space
        self.nu = 3
        self.control = list(np.arange(-(self.nu -1)/2, (self.nu -1)/2 +1))

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,autoscale_on=False, xlim=(-2,2), ylim=(-2,2))
        self.ax.grid()
        self.ax.axis('equal')
        plt.axis([-2, 2, -2, 2])

        self.line, = self.ax.plot([],[], 'o-', lw=2)
        self.time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
        self.time_text = self.ax.text(0.05, 0.8, '', transform=self.ax.transAxes)
        
        #Parameters 
        self.gamma = 0.9
        self.sigma = 2.5
        self.r = 0.001
        self.c = 1
        self.k = 1
        self.diff_threshold = 0.01
        
        #env parameters
        self.g = 9.8 #gravity 
        self.L = 1 #Length of the pendulum
        self.M = 1 #Mass 
        self.a = self.g / self.L
        self.b = self.c / (self.M * np.square(self.L))
        
    def computePfVxp(self, x, u):
        fxu = np.array([[x[1][0]],[self.a * np.sin(x[0])[0] - self.b*x[1][0] + u/(self.M*np.square(self.L))]]) 
        
        mean = x + fxu*self.dt
        mean = [float(mean[0]), float(mean[1])]
        cov = np.array([[self.sigma,0],[0,self.sigma]])
        cov = np.dot(cov,np.transpose(cov))*self.dt
        
        x_list = []
        value_list = []
        for key, value in self.states.items():
            list_state = [value[0][0],value[1][0]]
            x_list.append(list_state)
            
            list_value = self.value[key]
            value_list.append(list_value)
            
        prob_distribution = multivariate_normal.pdf(x_list, mean, cov)
        total_prob = sum(prob_distribution)
        prob_distribution = prob_distribution / total_prob
        #Pf*v(x')
        add = 0 
        for i in range(len(prob_distribution)):
            add += prob_distribution[i]*value_list[i]
        return add        
    
    def computeStageCost(self, x, u): #l(x,u)
        cost = 1 - np.exp(self.k*np.cos(x[0]) - self.k) + (self.r / 2)*np.square(u)
        return cost
    
    def value_iteration(self):
        value_diff = np.inf
        v_compare = {}
        
        #chosen state
        tmp0 = []
        tmp1 = []
        tmp2 = [] 
        
        it = 0
        while value_diff > self.diff_threshold:
            #Store the chosen states to plot over the episode 
            tmp0.append(float(self.value[(0.0,0.0)]))
            tmp1.append(float(self.value[(0.8727,-1.7453)]))
            tmp2.append(float(self.value[(4.5379,1.0472)]))
            
            it += 1
            print(it)
            for key, value in self.states.items(): 
            
                #Policy Improvement
                #compute three controls value
                state = value
                value_tmp_list = []
                for j in self.control:
                    value_tmp = self.computeStageCost(state, j) + self.gamma * self.computePfVxp(state, j)
                    value_tmp_list.append(value_tmp)
                
                #Policy Update
                self.policy[key] = int(self.control[int(np.argmin(value_tmp_list))])
                #Value Update
                v_compare[key] = self.value[key]
                self.value[key] = min(value_tmp_list)
                
            diff_list = []
            for key, value in self.states.items(): 
                diff = abs(v_compare[key] - self.value[key]) 
                diff_list.append(diff)
            

            value_diff = float(max(diff_list))
            print(value_diff)
        policy = self.policy 
        return policy, tmp0, tmp1, tmp2
    
    def computePf(self, x, u):
        fxu = np.array([[x[1][0]],[self.a * np.sin(x[0])[0] - self.b*x[1][0] + u/(self.M*np.square(self.L))]]) 
        
        mean = x + fxu*self.dt
        mean = [float(mean[0]), float(mean[1])]
        cov = np.array([[self.sigma,0],[0,self.sigma]])
        cov = np.dot(cov,np.transpose(cov))*self.dt
        
        x_list = []
        for key, value in self.states.items():
            list_state = [value[0][0],value[1][0]]
            x_list.append(list_state)
            
        prob_distribution = multivariate_normal.pdf(x_list, mean, cov)
        total_prob = sum(prob_distribution)
        prob_distribution = prob_distribution / total_prob #(360,)
        return prob_distribution 
     
    def policy_iteration(self):
        value_diff = np.inf
        #Initialize random policy for each state
        for key,value in self.policy.items():
            self.policy[key] = randint(-1,1)
            
        v = np.zeros((self.state_num,1))
        l = np.zeros((self.state_num,1))
        it = 0
        v_compare = v 
        
        #chosen state
        tmp0 = []
        tmp1 = []
        tmp2 = [] 
        while value_diff > self.diff_threshold:
            #Store the chosen states to plot over the episode 
            tmp0.append(float(self.value[(0.0,0.0)]))
            tmp1.append(float(self.value[(0.8727,-1.7453)]))
            tmp2.append(float(self.value[(4.5379,1.0472)]))
            
            print(it)
            it += 1
            #Policy Evaluation --> Solve the linear equation
            #Get the transition probability matrix 
            P = np.zeros((self.state_num ,self.state_num))
            i = 0 
            for key, value in self.states.items():
                distribution = self.computePf(value,self.policy[key])
                P[i] = distribution
                l[i] = self.computeStageCost(value,self.policy[key])
                i += 1
            inverse_term = np.linalg.inv(np.identity(self.state_num) - self.gamma * P)
            v = np.matmul(inverse_term,l)
            
            #Add the updated value into the self.value dictionary
            i = 0
            for key,value in self.states.items():
                self.value[key] = v[i] 
                i += 1
            
            #Policy Improvement
            for key,value in self.states.items(): 
                #compute three controls value
                state = value
                value_tmp_list = []
                for j in self.control:
                    value_tmp = self.computeStageCost(state, j) + self.gamma * self.computePfVxp(state, j)
                    value_tmp_list.append(value_tmp)
                #Policy Update
                self.policy[key] = int(self.control[int(np.argmin(value_tmp_list))])   
             
            #Compute largest difference    
            diff_list = []
            for i in range(len(v)): 
                diff = abs(v_compare[i] - v[i]) 
                diff_list.append(diff)
            value_diff = float(max(diff_list))
            v_compare = v 
            print(value_diff)
        
        policy = self.policy 
        return policy, tmp0, tmp1, tmp2
            
                    
    def computeNextState(self,xtuple,policy):
        brownian_noise1 = norm.rvs(scale=self.dt*0.25**2)
        brownian_noise2 = norm.rvs(scale=self.dt*0.25**2)
        x =  [[xtuple[0]+ xtuple[1]*self.dt + self.sigma*brownian_noise1], 
              [xtuple[1] + (self.a*np.sin(xtuple[0]) - self.b*xtuple[1] + policy[xtuple])*self.dt + self.sigma*brownian_noise2]]
        return x 
    
    def getNearestState(self,x):
        min_diff = np.inf
        for key, value in self.states.items(): 
            diff = (x[0][0]-value[0][0])**2 + (x[1][0]-value[1][0])**2
            if diff < min_diff:
                nearest_state = key
        return nearest_state 
    
    def getTraj(self,policy):
        traj =[]
        #Get Random State
        start_key, start_value = random.choice(list(self.states.items()))
        t = 0 
        while t < self.time: 
            next_state = self.computeNextState(start_key,policy)
            next_state[0][0] = next_state[0][0]%(2*np.pi)
            traj.append(next_state[0][0])
            start_key = self.getNearestState(next_state)
            t += self.dt
            print(t)
        print(traj)
        return traj 
            
            
    '''
    Provide new rollout theta values to reanimate
    '''
    def new_data(self, theta):
        self.theta = theta
        self.x1 = np.sin(theta)
        self.y1 = np.cos(theta)
        self.u = np.zeros(self.t.shape[0])

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,autoscale_on=False, xlim=(-2,2), ylim=(-2,2))
        self.ax.grid()
        self.ax.axis('equal')
        plt.axis([-2, 2,-2, 2])
        self.line, = self.ax.plot([],[], 'o-', lw=2)
        self.time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
        self.time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)

    def init(self):
        self.line.set_data([], [])
        self.time_text.set_text('')
        return self.line, self.time_text

    def _update(self, i):
        thisx = [0, self.x1[i]]
        thisy = [0, self.y1[i]]
        self.line.set_data(thisx, thisy)
        self.time_text.set_text(self.time_template % (self.t[i], self.theta[i], self.u[i]))
        return self.line, self.time_text

    def start(self):
        print('Starting Animation')
        # Set up plot to call animate() function periodically
        self.ani = FuncAnimation(self.fig, self._update, frames=range(len(self.x1)), interval=25, blit=True, init_func=self.init, repeat=True)
        plt.show()
        return self.states

if __name__ == '__main__':
    animation = EnvAnimate()
    
#    optimal_policy, tmp0, tmp1, tmp2 = animation.value_iteration()
    optimal_policy, tmp0, tmp1, tmp2 = animation.policy_iteration()

#    output1 = open('Policy_vi.pkl','wb')
    output1 = open('Policy_pi.pkl','wb')
    pickle.dump(optimal_policy,output1)
    output1.close()
    
#    output2 = open('tmp0_vi.pkl','wb')
    output2 = open('tmp0_pi.pkl','wb')
    pickle.dump(tmp0, output2)
    output2.close()
    
#    output3 = open('tmp1_vi.pkl','wb')
    output3 = open('tmp1_pi.pkl','wb')
    pickle.dump(tmp1, output3)
    output3.close()
    
#    output4 = open('tmp2_vi.pkl','wb')
    output4 = open('tmp2_pi.pkl','wb')
    pickle.dump(tmp2, output4)
    output4.close()
    
    traj = animation.getTraj(optimal_policy)
    animation.new_data(traj)
    state= animation.start()
