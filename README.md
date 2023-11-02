# CartPole Solution with Vanilla Policy Gradient
## A reinforcement learning agent trained using the Vanilla Policy Gradient algorithm to solve the CartPole problem from OpenAI's gym.
### Introduction
This project is a reflection of the work and knowledge I've acquired. For a detailed introduction and background on the CartPole problem, please refer to [Gymnasium](https://gymnasium.farama.org/environments/classic_control/cart_pole/).
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
3. Collect a set of trajectories by executing the current policy in the environment.
4. Compute the rewards-to-go as an estimate for Q^π(𝑠,𝑎).
5. Compute the policy gradient estimate using the rewards-to-go.
6. Update the policy parameters using some variant of gradient ascent.

### Licence
This project is licensed under the MIT License.
