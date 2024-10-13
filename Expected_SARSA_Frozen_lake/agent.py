import numpy as np

class Agent(object):
    """The world's simplest agent!"""
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.q_table = np.random.uniform(0,1,(self.state_space,self.action_space))
        self.q_table[-1:] = np.zeros((1,self.action_space))
        print(self.q_table)
        self.learning_rate =0.05
        self.exploration_rate= 1
        self.max_exploration_rate =1
        self.min_exploration_rate= 0.01
        self.exploration_decay_rate =0.001
        self.discount=0.95
        self.e = 0.6 # this is epsilon value to find weighted average of action in a state (to find expected value)

    def observe(self, observation, action, reward, done,old_state,truncated):
        #Add your code here to update q_table
        new_state  = observation
        new_action = self.act(observation)
        # if reward ==1:
        #     print("got rewar!")
        expected_q= 0 
        
        q_max = np.max(self.q_table[new_state,:])
        
        greedy_actions =0 
        tolerance = 1e-5
        for i in range(self.action_space):
            # print(self.q_table[new_state,:] , q_max )
            if np.isclose(self.q_table[new_state, i], q_max, atol=tolerance):
                greedy_actions += 1   
        
        non_greedy_act_prob = self.e / self.action_space
        greedy_act_prob = ((1 - self.e) / greedy_actions) + non_greedy_act_prob

        for i in range(self.action_space):
            if self.q_table[new_state,i] ==q_max:
                expected_q+=self.q_table[new_state,i] * greedy_act_prob
            else:
                expected_q += self.q_table[new_state,i] * non_greedy_act_prob
             
        if done:
            
            self.q_table[old_state, action] = self.q_table[old_state, action] + \
                                            self.learning_rate * (reward - self.q_table[old_state, action])
        else:
            self.q_table[old_state, action] = self.q_table[old_state, action] + \
                                      self.learning_rate * (reward + self.discount * (expected_q - self.q_table[old_state, action]))
            
        return new_action
    

        
    def act(self, observation):
        #Add your code here
        
        exploration_threshold = np.random.uniform(0,1)
        if exploration_threshold> self.exploration_rate: #exploitation
            return np.argmax(self.q_table[observation,:])
        else:
            return np.random.randint(self.action_space) #exploration
        


        

        