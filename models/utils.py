import numpy as np
import torch
from collections import deque
import gym
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output

class ExperienceBuffer:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.rewards = []
        self.log_probs = []
        
    def add(self, reward, log_prob):
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        
    def get_discounted_rewards(self, gamma=0.99):
        # 从后往前计算折扣奖励
        R = 0
        discounted_rewards = []
        
        for r in self.rewards[::-1]:
            R = r + gamma * R # R_t = r_t + gamma * R_{t+1}
            discounted_rewards.insert(0, R)
            
        # 标准化处理，使奖励均值为0，方差为1
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        return discounted_rewards
    
    def get_policy_loss(self, gamma=0.99):
        # 检查是否有足够的数据
        if len(self.rewards) == 0 or len(self.log_probs) == 0:
            return torch.tensor(0.0, requires_grad=True)
        
        discounted_rewards = self.get_discounted_rewards(gamma)
        log_probs = torch.stack(self.log_probs)
        policy_loss = -(log_probs * discounted_rewards).sum()
        
        return policy_loss

def plot_training_results(episode_rewards, window_size=100):
    """计算移动平均奖励"""
    moving_avg = deque(maxlen=window_size)
    moving_averages = []
    for reward in episode_rewards:
        moving_avg.append(reward)
        moving_averages.append(sum(moving_avg) / len(moving_avg))
    
    return moving_averages 



# 使用 matplotlib 展示训练后的智能体表现
def show_agent(env, policy, episodes=3):
    env_render = gym.make('CartPole-v1', render_mode='rgb_array')
    plt.figure(figsize=(8, 4))
    
    for episode in range(episodes):
        state, _ = env_render.reset()
        total_reward = 0
        
        while True:
            # 获取并显示当前帧
            frame = env_render.render()
            
            plt.clf()  # 清除当前图形
            plt.imshow(frame)
            plt.axis('off')
            plt.title(f'Episode {episode + 1}, Current Reward: {total_reward}')
            clear_output(wait=True)
            plt.show()
            
            action, _ = policy.select_action(state)
            try:
                result = env_render.step(action)
                if len(result) == 5:
                    state, reward, terminated, truncated, _ = result
                    done = bool(terminated) or bool(truncated)
                else:
                    state, reward, done, _ = result
                    done = bool(done)
            except Exception as e:
                print(f"Error during visualization: {e}")
                break
                
            total_reward += reward
            time.sleep(0.05)  # 添加小延迟使显示更流畅
            
            if done:
                print(f'Episode {episode + 1} finished with reward {total_reward}')
                break
                
    env_render.close()
    plt.close()
