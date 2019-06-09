import gym
import numpy as np
import pickle

class Planner:
    '''
	Initialization of all necessary variables to generate a policy:
		discretized state space
		control space
		discount factor
		learning rate
		greedy probability (if applicable)
	'''
    def __init__(self):
        #Initialize the Environment
        self.env = gym.make('MountainCar-v0')
        
        #Discretize the space = [x, v]
        env = self.env
        self.xmax = env.observation_space.high[0]
        self.xmin = env.observation_space.low[0]
        self.vmax = env.observation_space.high[1]
        self.vmin = env.observation_space.low[1]
        self.xresolution = 0.1
        self.xround = 1
        self.vresolution = 0.01
        self.vround = 2
        self.states = {} 
        self.xlen = int(round(((self.xmax - self.xmin) / self.xresolution),0))+1
        self.vlen = int(round(((self.vmax - self.vmin) / self.vresolution),0))+1
        for i in range(self.xlen):
            for j in range(self.vlen):
                self.states[(round((self.xmin + i * self.xresolution),self.xround) , round((self.vmin + j * self.vresolution),self.vround))] \
                    = ([round((self.xmin + i * self.xresolution),self.xround) , round((self.vmin + j * self.vresolution),self.vround)])
        
        self.state_num = len(self.states)
        
        #Initialize the Q value
        self.Q = {} 
        for i in range(self.xlen):
            for j in range(self.vlen):
                self.Q[(round((self.xmin + i * self.xresolution),self.xround) , round((self.vmin + j * self.vresolution),self.vround))] \
                = np.random.uniform(low = -1, high = 1, size=(self.env.action_space.n))

        
        #Control space
        #Use env.action_space
        self.control_num = 3
        
        #Parameters
        self.discount_factor = 0.9
        self.epsilon = 0.4
        self.episode_num = 12000
        self.cycle_num = 600
        self.learning_rate = 0.01
        
    '''
	Learn and return a policy via model-free policy iteration.
	'''
    def __call__(self, mc=False, on=True):
        return self._td_policy_iter(not on)


    '''
	TO BE IMPLEMENT
	TD Policy Iteration
	Flags: on : on vs. off policy learning
	Returns: policy that minimizes Q wrt to controls
	'''
    def _td_policy_iter(self, on=True):
        #Learning Cycle
        if on:
            print('SARSA')
        if not on: 
            print('Q-learning')
        
        tmp0 = []
        tmp1 = []
        tmp2 = []
        
        for i in range(self.cycle_num):
            greedy_policy = self.generateGreedyPolicy(self.Q)  
            episode = self.generateEpisode(greedy_policy)

            for j in range(self.episode_num):
                if j == self.episode_num - 1:
                    break
                curr_state, curr_control, curr_reward = episode[j]
                
                if curr_state[0] > 0.5:
                    break
                
                next_state, next_control, next_reward = episode[j+1]
                
                #on-policy learning: SARSA
                if on:
                    self.Q[tuple(curr_state)][curr_control] +=  self.learning_rate * (curr_reward + self.discount_factor * self.Q[tuple(next_state)][next_control] - self.Q[tuple(curr_state)][curr_control])
                    
                #off-policy learning: Q-learning
                if not on:
                    self.Q[tuple(curr_state)][curr_control] +=  self.learning_rate * (curr_reward + self.discount_factor * min(self.Q[tuple(next_state)]) - self.Q[tuple(curr_state)][curr_control])
           
            tmp0.append(self.Q[(0,0)])
            tmp1.append(self.Q[(-1.0,0.05)])
            tmp2.append(self.Q[(0.3,-0.05)])     
                    
            #Extract policy from Q 
            policy = {}
            for key, value in self.Q.items():
                policy[key] = np.argmax(value)
                
        return policy, self.Q, tmp0, tmp1, tmp2

    def computeStochasticPolicy(self,Qi):
        best = 1 - self.epsilon + (self.epsilon / self.control_num)
        other = (self.epsilon / self.control_num)
        if np.argmax(Qi) == 0:
            prob_distribution = [best, other, other]
        if np.argmax(Qi) == 1:
            prob_distribution = [other, best, other]
        if np.argmax(Qi) == 2: 
            prob_distribution = [other, other, best]
        return prob_distribution
    
    def generateGreedyPolicy(self,Q):
        greedy_policy = {}
        for key, value in Q.items():
            greedy_policy[key] = int(np.random.choice(self.control_num, 1, self.computeStochasticPolicy(value)))
        return greedy_policy       
                
    def generateEpisode(self,greedy_policy):
        env = self.env
        start = env.reset()
        n_state = [round(start[0],self.xround),round(start[1],self.vround)]
        episode = [] 
        episode.append([n_state, greedy_policy[tuple(n_state)],-1])
        
        for i in range(self.episode_num): 
            next_state, reward, done, _ = env.step(greedy_policy[tuple(n_state)])
            n_state = [round(next_state[0],self.xround),round(next_state[1],self.vround)]
            episode.append([n_state, greedy_policy[tuple(n_state)],reward])
            
        return episode
    
    '''
	Sample trajectory based on a policy
	'''
    def rollout(self, policy=None, render=True):
        traj = []
        t = 0
        done = False
        float_state = self.env.reset()
        c_state = [round(float_state[0],self.xround),round(float_state[1],self.vround)]
        if policy is None:
            while not done or t < 200:
                action = self.env.action_space.sample()
                if render:
                    self.env.render()
                n_state, reward, done, _ = self.env.step(action)
                traj.append((c_state, action, reward))
                c_state = n_state
                t += 1
            self.env.close()
            return traj

        else:
            while c_state[0] <= 0.5:

                action = policy[tuple(c_state)]
                if render:
                    self.env.render()

                n_state, reward, done, _ = self.env.step(action)
                traj.append((n_state, action, reward))
                float_state = n_state
                c_state = [round(float_state[0],self.xround),round(float_state[1],self.vround)]
                t += 1
            self.env.close()

            return traj


if __name__ == '__main__':
    planner = Planner()
    Policy, Qvalue, tmp0, tmp1, tmp2 = planner()
    
#    output1 = open('Policy_SA.pkl','wb')
    output1 = open('Policy_QL.pkl','wb')
    pickle.dump(Policy,output1)
    output1.close()
    
#    output2 = open('Qvalue_SA.pkl','wb')
    output2 = open('Qvalue_QL.pkl','wb')
    pickle.dump(Qvalue,output2)
    output2.close()
    
#    output3 = open('tmp0_SA.pkl','wb')
    output3 = open('tmp0_QL.pkl','wb')
    pickle.dump(tmp0,output3)
    output3.close()
    
#    output4 = open('tmp1_SA.pkl','wb')
    output4 = open('tmp1_QL.pkl','wb')
    pickle.dump(tmp1,output4)
    output4.close()
    
#    output5 = open('tmp2_SA.pkl','wb')
    output5 = open('tmp2_QL.pkl','wb')
    pickle.dump(tmp2,output5)
    output5.close()
    
    traj = planner.rollout(policy=Policy, render=True)
    print(traj)

    
        
        
        