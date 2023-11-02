import torch
from init_model import CustomModel
import torch.nn as nn

class Agent():
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, config, device):
        self.state_size = state_size
        self.action_size = action_size

        # Policy network and optimizer
        self.layers = config['model']['layers']
        self.policy = CustomModel(self.state_size, self.layers, self.action_size)
        self.seed = config['seed']
        self.lr = config['optimizer']['learning_rate']
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.update_every = config['update_every']
        self.layers = config['model']['layers']
        self.optimizer_type = config['optimizer']['type']
        self.device = device
        
        # init time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

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
        state = torch.from_numpy(state).float().unsqueeze(0)
        logits = self.policy(state)
        # Using softmax to convert logits to probabilities
        probs = nn.functional.softmax(logits, dim=1)
        action_distribution = torch.distributions.Categorical(probs)
        action = action_distribution.sample()

        # Storing the log probability of the action taken
        log_prob = action_distribution.log_prob(action)
        self.log_probs.append(log_prob)

        return action.item()

    def step(self, experiences):
        # Obtain random minibatch of tuples from D
        obs, reward, termination, truncation, info = experiences
        self.rewards.append(reward)

    def learn(self):
        # Calculate the discounted returns (future cumulative rewards)
        R = 0
        discounted_returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted_returns.insert(0, R)
        
        discounted_returns = torch.tensor(discounted_returns)
        discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-5)

        # Calculating the policy loss
        policy_loss = []
        for log_prob, R in zip(self.log_probs, discounted_returns):
            policy_loss.append(-log_prob * R)

        policy_loss = torch.cat(policy_loss).sum()

        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Clear the stored rewards and log probs for the next episode
        self.rewards = []
        self.log_probs = []

    # def soft_update(self, local_network, target_network, tau):
    #     """Soft update model parameters.
    #     θ_target = τ*θ_local + (1 - τ)*θ_target

    #     Params
    #     ======
    #         local_model (PyTorch model): weights will be copied from
    #         target_model (PyTorch model): weights will be copied to
    #         tau (float): interpolation parameter 
    #     """
    #     for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
    #         target_param.data.copy_(
    #             tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, model_name, model_path):
        self.policy.save(
            self.policy, model_name, model_path)

    def add_epoch_metrics(self, epoch_metrics):
        self.policy.add_epoch_metrics(epoch_metrics)

    def get_model_metrics(self):
        return self.policy.get_model_metrics()
            
        