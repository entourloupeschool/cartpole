import torch
from init_model import CustomModel
import torch.nn as nn

class AgentVPG():
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, config, device):
        self.state_size = state_size
        self.action_size = action_size

        # Policy network and optimizer
        self.layers = config['model']['layers']
        self.policy = CustomModel(self.state_size, self.layers, self.action_size)
        self.lr = config['optimizer']['learning_rate']
        self.gamma = config['gamma']
        self.optimizer_type = config['optimizer']['type']
        self.device = device

        self.optimizer = getattr(torch.optim, self.optimizer_type)(
            self.policy.parameters(), lr=self.lr)
        
        # Store rewards and log probs for each episode
        self.rewards = []
        self.log_probs = []

    def load_model(self, path):
        load = torch.load(path)
        
        # Load policy network
        self.policy.load_state_dict(load['model_state_dict'])
        
        # Load optimizer
        self.optimizer.load_state_dict(load['optimizer_state_dict'])
        
        self.policy.model_metrics = load['metrics']

        print("Model loaded from {}".format(path))

    def act(self, state):
        # Get action from policy network
        state = torch.from_numpy(state).float().unsqueeze(0) # Convert state to tensor
        logits = self.policy(state) # Forward pass
        # Using softmax to convert logits to probabilities
        probs = nn.functional.softmax(logits, dim=1) # softmax : probs = exp(logits) / sum(exp(logits))
        action_distribution = torch.distributions.Categorical(probs) # Categorical distribution : https://pytorch.org/docs/stable/distributions.html#torch.distributions.categorical.Categorical
        action = action_distribution.sample() # Sample an action from the distribution

        # Storing the log probability of the action taken
        log_prob = action_distribution.log_prob(action) # Log probability of the action taken
        self.log_probs.append(log_prob) # Store log probability for training

        return action.item()

    def step(self, experiences):
        # Obtain random minibatch of tuples from D
        obs, reward, termination, truncation, info = experiences
        self.rewards.append(reward)

    def learn(self):
        # Calculate the discounted returns (future cumulative rewards)
        R = 0 # Discounted return
        discounted_returns = [] # List of discounted returns
        for r in reversed(self.rewards): # Iterate in reverse order
            R = r + self.gamma * R # Update discounted return
            discounted_returns.insert(0, R) # Insert updated discounted return to the front
        
        discounted_returns = torch.tensor(discounted_returns) # Convert to tensor
        discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-5) # Normalize discounted returns

        # Calculating the policy loss
        policy_loss = [] # List to store the loss for each episode
        for log_prob, R in zip(self.log_probs, discounted_returns): # Iterate over the log probs and discounted returns
            policy_loss.append(-log_prob * R) # Append the loss to the list

        policy_loss = torch.cat(policy_loss).sum() # Concatenate the loss and sum them up

        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Clear the stored rewards and log probs for the next episode
        self.rewards = []
        self.log_probs = []

    def save(self, model_name, model_path):
        self.policy.save(
            self.policy, model_name, model_path)

    def add_epoch_metrics(self, epoch_metrics):
        self.policy.add_epoch_metrics(epoch_metrics)

    def get_model_metrics(self):
        return self.policy.get_model_metrics()
            
        