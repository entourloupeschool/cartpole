# environment install gymnasium[classic-control]
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import gym
import yaml
import torch
import datetime
import math
import numpy as np
from init_agent import AgentVPG
from plot_utils import plot_metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Load the configuration file
with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)
    
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print("Seed set to {}".format(seed))

    return seed

set_seed(config['seed'])
time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print("time: ", time)
model_id = 'best_model'+str(time)

if config['save_model']:
    # make folder in the models/ directory
    os.makedirs(config['model_save_path'] + '/' + model_id + '/')
    config['model_save_path'] = config['model_save_path'] + '/' + model_id + '/'
    print("Model will be saved at {}".format(config['model_save_path']))
    if config['save_results']:
    # make folder in the models/ directory
        os.makedirs(config['results_save_path'] + '/' + model_id + '/results/')
        config['results_save_path'] = config['results_save_path'] + '/' + model_id + '/results/'
        print("Results will be saved at {}".format(config['results_save_path']))
        
env = gym.make(config['env_name'], render_mode=config['render_mode'])
observation, info = env.reset(seed=config['seed'])
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
action_space = env.action_space

print("State size: ", state_size)
print("Action size: ", action_size)
print("Action space: ", action_space)

total_neurons = 0
# loop through the layers, and add the neurons
for layer in config['model']['layers']:
    total_neurons += layer[0]
print("Total number of neurons: {}".format(total_neurons))

# scale learning rate with number of neurons
config['optimizer']['learning_rate'] = config['optimizer']['learning_rate'] / \
    math.sqrt(total_neurons)
print("Learning rate: {}".format(config['optimizer']['learning_rate']))

# init agent
agent = AgentVPG(state_size, action_size, config, device)

def train_agent(agent, n_episodes=2000):
    observation, _ = env.reset()
    sums_rewards = []
    prints = 50
    episode_n = 1
    best_rewards = 0
    while episode_n < n_episodes:
        action = agent.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)

        agent.step((action, reward, observation, terminated, truncated))
        if terminated or truncated:
            obs, info = env.reset()
            s_r = sum(agent.rewards)
            sums_rewards.append(s_r)
            agent.add_epoch_metrics(
                {'episode': episode_n, 'total_reward': s_r})

            if episode_n % prints == 0:
                last_rewards = np.mean(sums_rewards[-prints:])
                print(f"Episode: {episode_n}, Rewards: {last_rewards}")
                if last_rewards > best_rewards:
                    best_rewards = last_rewards
                    if config['save_model']:
                        agent.save(model_id, config['model_save_path'])
                    if config['save_results']:
                        plot_metrics(agent.get_model_metrics(), model_id, config['save_results'], config['results_save_path'], config["show_plots"])
                        
            episode_n += 1
            agent.learn()
            
        env.render()
    env.close()
    
# train_agent(agent, n_episodes=config["number_of_episodes"])
