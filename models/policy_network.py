import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device='cpu'):
        super(PolicyNetwork, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # 添加一层
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.to(device)  # 将整个模型移动到指定设备
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=-1)
    
    def select_action(self, state):
        # 确保状态是正确的形状并移动到正确的设备
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # 添加batch维度
        state = state.to(self.device)
        
        probs = self.forward(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)
    def get_policy_loss(self, gamma=0.99):
        discounted_rewards = self.get_discounted_rewards(gamma)
        # 使用奖励均值作为基线
        baseline = discounted_rewards.mean()
        advantage = discounted_rewards - baseline
        log_probs = torch.stack(self.log_probs)
        policy_loss = -(log_probs * advantage).mean() 
