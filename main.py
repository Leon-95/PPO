import gym

from data_generator import DataGeneratorGymDiscrete
from models import Actor, Critic
from ppo import PPO

env = gym.make('CartPole-v1')
data_generator = DataGeneratorGymDiscrete(env)

critic = Critic(env.observation_space.shape[0])
actor = Actor(env.observation_space.shape[0], env.action_space.n)
algo = PPO(actor, critic)

for i in range(1000):
    print(i)
    policy = algo.get_policy()
    obs_batch, nobs_batch, act_batch, logprob_batch, rwd_batch, done_batch = data_generator.generate(5, policy)
    algo.train(obs_batch, nobs_batch, act_batch, logprob_batch, rwd_batch, done_batch)
    print("------------------")
