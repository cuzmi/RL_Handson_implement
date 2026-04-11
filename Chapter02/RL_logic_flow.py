"""
基于对强化学习最原始的理解, 来构建一个RL流程框架
env, agent -> action, reward -> new_env, agent

概念封装 和 交互对象封装 // 说实话, 这些方式没有多大的区别, 只要能正确表现出其逻辑流程即可
"""
import random
# 上面的demo 是以概念为核心的，action reward observation这几个概念分类到env和agent里面的
class Env():
    def __init__(self, init_obs):
        self.observation = init_obs
    
    def get_observation(self):
        return [1.0, 1.0, 1.0]
    
    def reward(self, observation, action):
        return 1.0
    
    def is_done(self):
        return False

class Agent():
    def __init__(self):
        self.reward = 0

    def action(self, observation):
        return [1.0, 0.0]
    

# ===========================
# 下面则是交互协议的思想 agent部分只给出action，然后接受来自env的内容
        
class Env():
    def __init__(self):
        self.steps = 10
    
    def get_observation(self):
        return [1.0, 1.0, 0.0]
    
    def get_actions(self, observation):
        return 
    
    def is_done(self):
        return self.steps == 0
    
    def action(self, action):
        # 这部分就是agent选择动作后 env内部的变化了
        if self.is_done():
            raise Exception("Game is over")
        self.steps -= 1
        return 

class Agent():
    def __init__(self):
        self.total_reward = 0.0
    
    def step(self, env):
        # 都是从env就收信息，只有中间的action是自己选择的 / 把step放到agent里面是因为是agent的动作推进下一步
        # 把step放到agent里面，是把agent当成一个完整的决策封装单元，而不是单一的策略
        # 如果把agent当成一个独立的策略部分，那么step可以放到main流程里面，其实是一样的, 甚至于放到env里面都没问题
        obs = env.get_observation()
        actions = env.get_actions()
        reward = env.action(random.choice(actions))
        self.total_reward += reward
    
if __name__ == "__main__":
    env = Env()
    agent = Agent()
    if env.is_done():
        # 这里也可以把agent看成一个更大的完整测试封装单元
        agent.step(env)