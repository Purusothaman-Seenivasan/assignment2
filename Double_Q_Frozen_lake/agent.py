import numpy as np

class Agent(object):
    """The world's simplest agent!"""
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.q_table = np.random.uniform(0,0.1,(self.state_space,self.action_space))
        self.q_table[-1:] = np.zeros((1,self.action_space))
        self.q_table2 = np.random.uniform(0,0.1,(self.state_space,self.action_space))
        self.q_table2[-1:] = np.zeros((1,self.action_space))
        print(self.q_table , self.q_table2)
        self.learning_rate =0.05
        self.exploration_rate= 1
        self.max_exploration_rate =1
        self.min_exploration_rate= 0.01
        self.exploration_decay_rate =0.001
        self.discount=0.95

    def observe(self, observation, action, reward, done,old_state,truncated):
        #Add your code here to update q_table
        new_state  = observation
        # if reward ==1:
        #     print("got rewar!")
        if done:  
            if np.random.rand() < 0.5:  
                self.q_table[old_state, action] += self.learning_rate * (reward - self.q_table[old_state, action])
            else:
                self.q_table2[old_state, action] += self.learning_rate * (reward - self.q_table2[old_state, action])
        else:  
            if np.random.rand() < 0.5: 
                max_action = np.argmax(self.q_table[new_state, :])
                self.q_table[old_state, action] += self.learning_rate * (reward + self.discount * self.q_table2[new_state, max_action] - self.q_table[old_state, action])
            else:  
                max_action = np.argmax(self.q_table2[new_state, :])
                self.q_table2[old_state, action] += self.learning_rate * (reward + self.discount * self.q_table[new_state, max_action] - self.q_table2[old_state, action])

    def act(self, observation,explore_vs_exploit,status):
        #Add your code here
        
        exploration_threshold = np.random.uniform(0,1)
        if exploration_threshold> self.exploration_rate: #exploitation
            if status != "Test":
                explore_vs_exploit["exploitation"]+=1
            return np.argmax((self.q_table[observation,:] +self.q_table2[observation,:]))
        
        

        else:
            if status != "Test":
                explore_vs_exploit["exploration"]+=1
            return np.random.randint(self.action_space) #exploration
        


        

        