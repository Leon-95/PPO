import numpy as np
import torch
import random


class DataGeneratorGymDiscrete:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cpu")

        self.obs_size = self.env.observation_space.shape[0]
        self.act_size = self.env.action_space.n

        # Define replay pool
        self.buffer = []
        self.idx = 0

    def get_action(self, policy, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        act, logprob = policy(obs)
        return act, logprob

    def generate(self, episodes, policy):
        obs_batch = []
        nobs_batch = []
        act_batch = []
        logprob_batch = []
        rwd_batch = []
        done_batch = []

        avg_rwd = 0.0
        for i in range(episodes):
            done = False
            obs = self.env.reset()

            ep_rwd = 0.0
            while not done:
                #self.env.render()
                act, logprob = self.get_action(policy, obs)
                nobs, rwd, done, _ = self.env.step(act)

                obs_batch.append(obs)
                nobs_batch.append(nobs)
                act_batch.append(act)
                logprob_batch.append(logprob)
                rwd_batch.append(rwd)
                done_batch.append(done)

                obs = nobs
                ep_rwd += rwd

            avg_rwd += ep_rwd

        avg_rwd /= episodes
        print("Reward " + str(avg_rwd))
        return np.array(obs_batch), np.array(nobs_batch), np.array(act_batch), np.array(logprob_batch), np.array(rwd_batch), np.array(done_batch)
