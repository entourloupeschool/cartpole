# CartPole Solution with Vanilla Policy Gradient
## A reinforcement learning agent trained using the Vanilla Policy Gradient algorithm to solve the CartPole problem from OpenAI's gym.
### Introduction
For a detailed introduction and background on the CartPole problem, please refer to [Gymnasium](https://gymnasium.farama.org/environments/classic_control/cart_pole/).
![Alt text](https://cdn-images-1.medium.com/v2/resize:fit:1200/1*oMSg2_mKguAGKy1C64UFlw.gif)
### Vanilla Policy Gradient
The Vanilla Policy Gradient (VPG), also known as REINFORCE, is a foundational algorithm in the realm of policy optimization methods. At its heart, VPG seeks to maximize the expected return by directly adjusting the agent's policy. The core idea revolves around increasing the probability of actions that result in higher returns and decreasing the probability of actions leading to lower returns.

In practice, VPG calculates the gradient of the expected return concerning the policy parameters and then applies gradient ascent. One of VPG's significant traits is its utilization of the entire trajectory, which aids in estimating the policy gradient. However, it's worth noting that while VPG can be effective, it's often perceived as having high variance, which can sometimes affect its stability and efficiency.

For a deeper dive into the mathematical intricacies and nuances of the VPG algorithm, refer to OpenAI's detailed documentation.

Here is a correlation of the VPG's pseudocode and the code in the agent's class:
1. Initialize policy parameters θ
   set up the policy network: The weights of this neural network are initialized with small random values, close to zero.
   ``` python
   self.policy = CustomModel(self.state_size, self.layers, self.action_size)
   ```
2. Collect a set of trajectories by executing the current policy in the environment. It first converts the input state to a PyTorch tensor and processes it through the policy network to get the logits.
    The logits are transformed to action probabilities using the softmax function.
    An action is then sampled from the resulting categorical distribution and returned.
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
   Whenever the agent interacts with the environment, it logs the reward it received for the action it took.
   ``` python
   def step(self, experiences):
      obs, reward, termination, truncation, info = experiences
      self.rewards.append(reward)
   ```
3. Compute the rewards-to-go as an estimate for Q^π(𝑠,𝑎).
   In the learn method, it calculates the discounted cumulative rewards (rewards-to-go). These are essential for estimating the returns of an action given a state. It starts from the last reward and moves backward, computing the accumulated discounted reward for each time step. The returns are also normalized to have a mean of zero and a standard deviation of one. This normalization often helps stabilize the learning.
   ``` python
   R = 0 # Discounted return
   discounted_returns = [] # List of discounted returns
   for r in reversed(self.rewards): # Iterate in reverse order
      R = r + self.gamma * R # Update discounted return
      discounted_returns.insert(0, R) # Insert updated discounted return to the front
   discounted_returns = torch.tensor(discounted_returns) # Convert to tensor
   discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-5) # Normalize discounted returns
   ```
4. Compute the policy gradient estimate using the rewards-to-go. This computes the policy gradient by iterating over the stored log probabilities and the computed discounted returns. Each episode's loss is calculated as the product of the negative log probability of the taken action and its discounted return. The objective is to increase the log probabilities of actions that led to good outcomes and decrease those that led to poor outcomes.
   ``` python
   # Calculating the policy loss
   policy_loss = [] # List to store the loss for each episode
   for log_prob, R in zip(self.log_probs, discounted_returns): # Iterate over the log probs and discounted returns
      policy_loss.append(-log_prob * R) # Append the loss to the list

   policy_loss = torch.cat(policy_loss).sum() # Concatenate the loss and sum them up
   ```   
5. Update the policy parameters using some variant of gradient ascent. This is performed with the optimizer, which updates the policy network's weights:
   ``` python
   self.optimizer.zero_grad()
   policy_loss.backward()
   self.optimizer.step()
   ```   

### Licence
This project is licensed under the MIT License.
