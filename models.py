
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(obs_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, obs):
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)
        return output


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(obs_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, act_dim)

    def forward(self, obs):
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = F.softmax(self.layer3(activation2), dim=-1)
        return output



