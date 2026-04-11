"""
对 hands on中 wrapper, actionwrapper, observationwrapper, rewardwrapper, monitor 
写代码 理解这四个内容之间的联系 & 分别的应用场景 - 实践
"""
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SimpleCounterEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # TODO 1:
        # 定义 action_space
        # 提示：只有两个离散动作 0 和 1
        self.action_space = None

        # TODO 2:
        # 定义 observation_space
        # 提示：state 是一个 shape=(1,) 的整数/浮点都行
        # 取值范围至少要覆盖 -3 到 3
        self.observation_space = None

        self.state = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # TODO 3:
        # 重置 state 为 0
        self.state = 0

        # TODO 4:
        # 返回 observation 和 info
        obs = None
        info = {}

        return obs, info

    def step(self, action):
        # TODO 5:
        # 根据 action 更新 state
        # action == 0 -> state -= 1
        # action == 1 -> state += 1

        # TODO 6:
        # 按规则计算 reward / terminated
        reward = 0
        terminated = False

        truncated = False
        info = {}

        # TODO 7:
        # 构造 observation
        obs = None

        return obs, reward, terminated, truncated, info


class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.2):
        super().__init__(env)
        self.epsilon = epsilon

    def action(self, action):
        # TODO 8:
        # 以 epsilon 概率，把 action 替换成随机动作
        # 否则返回原 action
        return action


class EpisodeStatsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episode_reward = 0
        self.episode_length = 0

    def reset(self, **kwargs):
        # TODO 9:
        # 每次开新 episode 时，把统计清零
        self.episode_reward = 0
        self.episode_length = 0

        return self.env.reset(**kwargs)

    def step(self, action):
        # TODO 10:
        # 调用原环境的 step
        obs, reward, terminated, truncated, info = self.env.step(action)

        # TODO 11:
        # 更新 episode_reward 和 episode_length

        # TODO 12:
        # 如果 episode 结束，把统计写入 info["episode"]
        # 格式例如：
        # info["episode"] = {"r": ..., "l": ...}

        return obs, reward, terminated, truncated, info


def run_demo():
    env = SimpleCounterEnv()
    env = RandomActionWrapper(env, epsilon=0.2)
    env = EpisodeStatsWrapper(env)

    obs, info = env.reset()
    print("reset ->", obs, info)

    done = False
    while not done:
        action = 1   # 先一直往右走，方便观察 wrapper 有没有随机扰动
        print("agent action =", action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        print(
            f"obs={obs}, reward={reward}, terminated={terminated}, info={info}"
        )

    print("episode finished")


if __name__ == "__main__":
    run_demo()