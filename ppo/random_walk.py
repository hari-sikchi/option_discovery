import gym
from gym import spaces
import numpy as np


class RandomWalkEnv(gym.Env):
    def __init__(self,total_states = '100'):
       self.past_actions = []
       self.metadata={'render.modes': ['human']}
       self.states = np.arange(total_states)

  
       self.reset()
       self.action_space = spaces.Discrete(2)
       self.observation_space = spaces.Discrete(1)
       self.cum_rewards = 0
       self.observation_dim=1
       self.action_dim = 2

    def get_int_state(self,state):
        if(state=='s1'):
            return 1
        elif(state=='s2'):
            return 2
        elif(state=='s3'):
            return 3
        elif(state=='dead'):
            return 4
        else:
            return -1


    def reset(self):
        # self.task = self.tasks[np.random.choice(len(self.tasks))]
        # print("New task is {}".format(self.task))
        # print("reset called")
        self.state = 0
        self.cum_rewards = 0
        return self.state

    def step(self,action):
        # print("Env state:{}",self.state)
        reward = 0
        done=False
        
        if(self.state == 0 and action==0):
            self.state = state
            reward = 0
        elif (self.state==self.total_states-1 and action==1):
            self.state = self.total_states+1
            done=True
            reward=1000
        elif action==0:
            self.state=self.state-1
        elif action==1:
            self.state=self.state+1
            
        self.cum_rewards +=reward
        if(done):
            return self.state,reward,done,{'episode':{'r':self.cum_rewards}}
        return self.state,reward,done,{}
        
 