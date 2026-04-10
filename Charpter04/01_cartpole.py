import numpy as np
import gymnasium as gym
from dataclasses import dataclass
from typing import List, Generator, Tuple
from torch.utils.tensorboard.writer import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70

class Net(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
@dataclass
class EpisodeStep:
    observation: np.ndarray
    action: int

@dataclass
class Episode:
    reward: float
    steps: List[EpisodeStep]

def iterate_batches(env: gym.Env, net: Net, batch_size: int) -> Generator[List[Episode], None, None]:
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs, _ = env.reset()
    sm = nn.Softmax(dim = 1)
    while True:
        # NOTE: 观测- 动作 - 观测 - [By: Weijie] - 2026/04/10
        obs_v = torch.tensor(obs, dtype = torch.float32)
        act_probs_v = sm(net(obs_v.unsqueeze(0)))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p = act_probs)
        next_obs, reward, is_done, is_trunc, _ = env.step(action)
        episode_reward += float(reward)
        step = EpisodeStep(observation = obs, action = action)
        episode_steps.append(step)
        if is_done or is_trunc:
            # NOTE: 记录 + 重置 - [By: Weijie] - 2026/04/10
            e = Episode(reward = episode_reward, steps = episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        
        obs = next_obs

def filter_batch(batch: List[Episode], percentile: float) -> Tuple[torch.FloatTensor, torch.LongTensor, float, float]:
    """
    
    Args:
        batch (List[Episode])
        percentile (float)

    Returns:
        Tuple[torch.FloatTensor, torch.LongTensor, float, float]
    
    
    """
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = float(np.percentile(rewards, percentile))
    reward_mean = float(np.mean(rewards))

    train_obs: List[np.ndarray] = []
    train_act: List[int] = []
    for episode in batch:
        if episode.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, episode.steps))
        train_act.extend(map(lambda step: step.action, episode.steps))
    
    train_obs_v = torch.FloatTensor(np.vstack(train_obs))
    train_act_v = torch.LongTensor(train_act)

    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    assert env.observation_space.shape is not None

    obs_size = env.observation_space.shape[0]
    assert isinstance(env.action_space, gym.spaces.Discrete)

    n_actions = int(env.action_space.n)

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    print(net)

    objective = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr = 0.01)
    writer = SummaryWriter(comment = "-cartpole")

    # TODO: 这部分是什么意思， 如何根据特定的data 更新的？ - [By: Weijie] - 2026/04/10
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print(f'{iter_no}: loss = {loss_v.item():.3f}, reward_mean = {reward_m:.1f}, reward_bound = {reward_b:.1f}')
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)

        if reward_m > 475:
            print("Solved!")
            break

        writer.close()