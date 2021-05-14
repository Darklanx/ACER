# -*- coding: utf-8 -*-
from torch import nn
from torch.nn import functional as F
import torch as th
import numpy as np


class ActorCritic(nn.Module):
  def __init__(self, observation_space, action_space, hidden_size):
    super(ActorCritic, self).__init__() 
    # print(observation_space.shape)
    n_input_channels = observation_space.shape[2]
    self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
    
    self.state_size = self.cnn(th.as_tensor(self.reorder_image(observation_space.sample())[None]).float()).shape[1]
    # exit()
    # self.state_size = observation_space.shape[0]
    self.action_size = action_space.n

    self.fc1 = nn.Linear(self.state_size, hidden_size)
    self.lstm = nn.LSTMCell(hidden_size, hidden_size)
    self.fc_actor = nn.Linear(hidden_size, self.action_size)
    self.fc_critic = nn.Linear(hidden_size, self.action_size)


  def reorder_image(self, observation):
      return np.moveaxis(observation, 2, 0)

  def forward(self, x, h):
    print("forward")
    x = self.reorder_image(x)
    x = self.cnn(x)
    x = F.relu(self.fc1(x))
    h = self.lstm(x, h)  # h is (hidden state, cell state)
    x = h[0]
    theta_s = self.fc_actor(x)
    policy = F.softmax(theta_s, dim=1).clamp(max=1 - 1e-20)  # Prevent 1s and hence NaNs
    Q = self.fc_critic(x)
    V = (Q * policy).sum(1, keepdim=True)  # V is expectation of Q under π
    return policy, Q, V, h, theta_s

