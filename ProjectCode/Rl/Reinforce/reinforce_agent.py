
import torch
import torch.optim as optim
from policy_network import PolicyNetwork

class REINFORCEAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        action_probs = self.policy_network(state)
        distribution = torch.distributions.Categorical(action_probs)
        action = distribution.sample()
        self.log_probs.append(distribution.log_prob(action))
        return action.item()

    def update_policy(self):
        R = 0
        policy_loss = []
        returns = []
        # Calculate the discounted future rewards
        for r in reversed(self.rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        # Calculate policy loss
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum() # changed cat to stack
        policy_loss.backward()
        self.optimizer.step()
        # Reset rewards and log probabilities
        self.log_probs = []
        self.rewards = []