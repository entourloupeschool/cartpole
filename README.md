# CartPole Solution with Vanilla Policy Gradient
## A reinforcement learning agent trained using the Vanilla Policy Gradient algorithm to solve the CartPole problem from OpenAI's gym.
### Introduction
For a detailed introduction and background on the CartPole problem, please refer to [Gymnasium](https://gymnasium.farama.org/environments/classic_control/cart_pole/).
### Vanilla Policy Gradient
The Vanilla Policy Gradient (VPG), also known as REINFORCE, is a foundational algorithm in the realm of policy optimization methods. At its heart, VPG seeks to maximize the expected return by directly adjusting the agent's policy. The core idea revolves around increasing the probability of actions that result in higher returns and decreasing the probability of actions leading to lower returns.

In practice, VPG calculates the gradient of the expected return concerning the policy parameters and then applies gradient ascent. One of VPG's significant traits is its utilization of the entire trajectory, which aids in estimating the policy gradient. However, it's worth noting that while VPG can be effective, it's often perceived as having high variance, which can sometimes affect its stability and efficiency.

For a deeper dive into the mathematical intricacies and nuances of the VPG algorithm, refer to OpenAI's detailed documentation.

Here is a correlation of the VPG's pseudocode and the code in the agent's class:
1. Initialize policy parameters θ
   set up the policy network: weights are initialized with random values close to 0.
   ``` python
   self.policy = CustomModel(self.state_size, self.layers, self.action_size)
   ```
3. Collect a set of trajectories by executing the current policy in the environment. This is achieved in the act method where actions are determined based on the current policy:
   ``` python
   def act(self, state):
    # Get action from policy network
    state = torch.from_numpy(state).float().unsqueeze(0) # Convert state to tensor
    logits = self.policy(state) # Forward pass
    # Using softmax to convert logits to probabilities
    probs = nn.functional.softmax(logits, dim=1) # softmax : probs = exp(logits) / sum(exp(logits))
    action_distribution = torch.distributions.Categorical(probs) # Categorical distribution : https://pytorch.org/docs/stable/distributions.html#torch.distributions.categorical.Categorical
    action = action_distribution.sample()
    return action.item()
   ```
   The step method logs the rewards for the chosen actions:
   ``` python
   def step(self, experiences):
    obs, reward, termination, truncation, info = experiences
    self.rewards.append(reward)
   ```
4. Compute the rewards-to-go as an estimate for Q^π(𝑠,𝑎).
   This is done in the learn method, where the discounted cumulative rewards are computed:
   ``` python
   R = 0 # Discounted return
   discounted_returns = [] # List of discounted returns
   for r in reversed(self.rewards): # Iterate in reverse order
    R = r + self.gamma * R # Update discounted return
    discounted_returns.insert(0, R) # Insert updated discounted return to the front
   discounted_returns = torch.tensor(discounted_returns) # Convert to tensor
   discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-5) # Normalize discounted returns
   ```
5. Compute the policy gradient estimate using the rewards-to-go.
   
6. Update the policy parameters using some variant of gradient ascent.

### Licence
This project is licensed under the MIT License.
