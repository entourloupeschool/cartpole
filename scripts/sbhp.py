# in this file we grid search for the best hyperparameters for the model
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import gym
import yaml
import torch
import datetime
import math
from init_agent import AgentVPG
from train_agent import train_agent
from rounding import truncate

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

# Hyperparameters
for lr_i in config['lr_search']:
    config['optimizer']['learning_rate'] = truncate(lr_i / math.sqrt(total_neurons), 5)
    for gamma_i in config['gamma_search']:
        config['gamma'] = truncate(gamma_i, 5)
        print( "lr: {}, gamma: {}".format(config['optimizer']['learning_rate'], config['gamma']) )
        # Initialize the agent
        agent = AgentVPG(state_size=state_size, action_size=action_size, config=config, device=device)
        # Train the agent
        train_agent(agent, config['sbhp_n_episodes'])
        del agent