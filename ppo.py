import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

lr_actor = 0.0001
lr_critic = 0.0001
gamma = 0.99
clip = 0.1
updates_per_iter = 10


class PPO(object):
    def __init__(self, actor, critic):
        self.device = torch.device("cpu")
        torch.manual_seed(123)
        np.random.seed(123)

        # Define actor and critic
        self.actor = actor
        self.critic = critic
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_critic)

    def get_policy(self):
        def policy(obs):
            probs = self.actor(obs).detach().numpy()
            act = np.random.choice(list(range(probs.shape[0])), p=probs)
            return act, np.log(probs[act])
        return policy

    def train(self, obs_batch, nobs_batch, act_batch, logprob_batch, rwd_batch, done_batch):
        # Compute returns
        ret_batch = []
        disc_rwd = 0.0
        for rwd, done in zip(reversed(rwd_batch), reversed(done_batch)):
            if done:
                disc_rwd = 0.0
            disc_rwd = rwd + gamma * disc_rwd
            ret_batch.insert(0, disc_rwd)
        ret_batch = np.array(ret_batch)

        # Convert everything to tensors
        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        act_batch = torch.FloatTensor(act_batch).to(self.device)
        logprob_batch = torch.FloatTensor(logprob_batch).to(self.device)
        rwd_batch = torch.FloatTensor(rwd_batch).to(self.device)
        ret_batch = torch.FloatTensor(ret_batch).to(self.device)
        nobs_batch = torch.FloatTensor(nobs_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        for _ in range(updates_per_iter):
            # Compute state values, log probabilites and advantages
            values = self.critic(obs_batch).squeeze()
            cur_probs = self.actor(obs_batch)
            cur_log_probs = torch.log(torch.gather(cur_probs, dim=1, index=act_batch.type(torch.LongTensor).unsqueeze(1)).squeeze())
            advantages = ret_batch - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
            ratios = torch.exp(cur_log_probs - logprob_batch)

            # Update actor
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - clip, 1 + clip) * advantages
            actor_loss = (-1 * torch.min(surr1, surr2)).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optim.step()

            # Update critic
            critic_loss = nn.MSELoss()(values, ret_batch)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
