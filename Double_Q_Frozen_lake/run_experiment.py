import argparse
import gymnasium as gym
import importlib.util
import numpy as np
import time
from matplotlib import pyplot as plt
from IPython.display import clear_output

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="agent.py")
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)




try:
    env = gym.make(args.env)
    print("Loaded ", args.env)
except:
    file_name, env_name = args.env.split(":")
    gym.envs.register(
        id=env_name + "-v0",
        entry_point=args.env,
    )
    env = gym.make(env_name + "-v0")
    print("Loaded", args.env)


#-----Paramters---------
num_episodes =20000
iteration =1
max_steps_per_episode=200
learning_rate= 0.1
discount=0.99

exploration =1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01
# ---------end of parameters
action_dim = env.action_space.n
state_dim = env.observation_space.n

agent = agentfile.Agent(state_dim, action_dim)

reward_list =np.zeros((num_episodes,iteration))

explore_vs_exploit ={"exploration":0 , "exploitation":0}
for i in range(iteration):

    for episode in range(num_episodes):
        observation,_ = env.reset()
        rewards_current_epi = 0
        done =False
        


        for stpes in range(max_steps_per_episode): 
            #env.render()
            

        
            action = agent.act(observation,explore_vs_exploit,status = "train") # your agent here (this currently takes random actions)
            old_state= observation
            observation, reward, done, truncated, info = env.step(action)
        
            rewards_current_epi+=reward
            agent.observe(observation,action, reward, done,old_state,truncated)#action added
            
            if done:
                observation, info = env.reset() 
                break 
        
        reward_list[episode,i] = rewards_current_epi


        agent.exploration_rate = agent.min_exploration_rate + \
                            (agent.max_exploration_rate - agent.min_exploration_rate) * \
                            np.exp(-agent.exploration_decay_rate * episode)
    # print(agent.exploration_rate)
env.close()

avg_reward = np.mean(reward_list,axis=1)
std_reward = np.std(reward_list, axis=1)
rewards_per_100episode = np.split(np.array(avg_reward),num_episodes/1000)
count =1000
print("avg reward per 1000 epi")
for r in rewards_per_100episode:
    print(count, ":",str(sum(r/1000)))
    count +=1000
print(agent.q_table)

print(agent.exploration_rate)


gym.make("FrozenLake-v1", render_mode="rgb_array")
for episode in range(3):
    observation = env.reset()[0]
    done = False
    print("*****EPISODE ", episode+1, "*****\n\n\n\n")
    time.sleep(1)
    for step in range(max_steps_per_episode):        
        clear_output(wait=True)
        print(env.render())
        time.sleep(0.3)
        action = agent.act(observation,explore_vs_exploit,status="Test")
        observation, reward, done, truncated, info = env.step(action)

        if done:
            clear_output(wait=True)
            print(env.render())
            if reward == 1:
                print("****You reached the goal!****")
                time.sleep(3)
            else:
                print("****You fell through a hole!****")
                time.sleep(3)
                clear_output(wait=True)
            break
env.close()
print("exploreation=",explore_vs_exploit["exploration"]/(num_episodes*iteration))
print("exploitation=",explore_vs_exploit["exploitation"]/(num_episodes*iteration))

reward_cumsum = np.cumsum(avg_reward)

running_average = reward_cumsum / np.arange(1, num_episodes + 1)
std_reward_running = std_reward / np.sqrt(np.arange(1, num_episodes + 1))

# Plotting with error bars
plt.errorbar(np.arange(0, num_episodes), reward_cumsum, yerr=std_reward_running, fmt='o', capsize=5, capthick=2, elinewidth=1)
plt.title('Cumulative Reward with Error Bars for Frozen lake')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.grid(True)
plt.show()


