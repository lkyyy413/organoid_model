import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from initialization_singlelayer import Initialization
from initialization_spheresheet import Initialization_spheresheet
from parthex_vertex_model_singlelayer import VertexModel
from part2_RD_VM_singlelayer import CoupledSimulator

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=400, fc2_dims=300):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        #x = torch.tanh(self.mu(x)) * 10
        x = self.mu(x)
        #x = torch.tanh(self.mu(x))  # 输出范围[-1, 1]
        return x

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=400, fc2_dims=300):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q(x)
        return q_value

class TD3Agent:
    def __init__(self, coupled_simulator, target_shape, model_path=None):
        self.simulator = coupled_simulator
        self.target_shape = target_shape
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 状态维度：顶点坐标
        self.state_dims = self._extract_state().shape[0]
        self.action_dims = 2  # 调整方程中的衰减率或生长率
        
        # 初始化网络
        self.actor = ActorNetwork(self.state_dims, self.action_dims).to(self.device)
        self.critic1 = CriticNetwork(self.state_dims, self.action_dims).to(self.device)
        self.critic2 = CriticNetwork(self.state_dims, self.action_dims).to(self.device)
        
        # 目标网络
        self.target_actor = ActorNetwork(self.state_dims, self.action_dims).to(self.device)
        self.target_critic1 = CriticNetwork(self.state_dims, self.action_dims).to(self.device)
        self.target_critic2 = CriticNetwork(self.state_dims, self.action_dims).to(self.device)
        
        # 同步目标网络参数
        self.update_target_networks(tau=1.0)
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=1e-3)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=1e-3)
        
        # 经验回放
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005
        self.update_delay = 2
        self.update_counter = 0
        
        # 探索噪声
        self.noise_sigma = 0.1
        self.noise_clip = 0.5
        
        # 加载预训练模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def _extract_state(self):
        """提取状态特征：仅顶点坐标"""
        vertices = self.simulator.vm.vertices.flatten().to(self.device)
        print("Vertices range:", vertices.min(), vertices.max())
        return vertices

    def _compute_reward(self):
        """奖励函数：仅基于形状相似度"""
        current_vertices = self.simulator.vm.vertices
        shape_reward = -torch.cdist(current_vertices, self.target_shape).mean()
        return shape_reward

    def choose_action(self, state, exploration=True):
        """选择动作，可选是否添加探索噪声"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).squeeze(0)
        
        if exploration:
            noise = torch.randn_like(action) * self.noise_sigma
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            #action = torch.clamp(action + noise, -1, 1)
            action = action + noise
        return action.detach().cpu().numpy()

    def apply_action(self, action):
        """应用动作调整方程中的衰减率或生长率"""
        # 假设模拟器有这两个参数可以调整
        # 这里需要根据实际模拟器的接口进行调整
        # 动作值在[-1,1]范围内，直接作为调整量
        self.simulator.rd.growth_rate_u += action[0]  # 调整U方程的生长率
        self.simulator.rd.decay_rate_v += action[1]   # 调整V方程的衰减率

    def store_experience(self, state, action, reward, next_state, done):
        """存储经验到回放池"""
        self.memory.append((state, action, reward, next_state, done))

    def update_target_networks(self, tau):
        """软更新目标网络"""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        # 从回放池采样
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        
        # 更新Critic
        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            target_actions += torch.clamp(torch.randn_like(target_actions) * 0.2, -0.5, 0.5)
            #target_actions = torch.clamp(target_actions, -1, 1)
            target_actions = target_actions
            
            target_q1 = self.target_critic1(next_states, target_actions)
            target_q2 = self.target_critic2(next_states, target_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # 延迟更新Actor和目标网络
        self.update_counter += 1
        if self.update_counter % self.update_delay == 0:
            actions = self.actor(states)
            actor_loss = -self.critic1(states, actions).mean()
        
            # 增加动作幅度的 L2 惩罚
            action_penalty = 0.01 * torch.mean(actions ** 2)  # 系数可调整
            actor_loss += action_penalty
        
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        
            self.update_target_networks(self.tau)

    def train(self, episodes=1000, steps_per_episode=5, save_path=None):
        episode_rewards = []
        for episode in range(episodes):
            state = self._extract_state().cpu().numpy()
            episode_reward = 0
            
            for step in range(steps_per_episode):
                action = self.choose_action(state)
                self.apply_action(action)
                
                self.simulator.step()
                
                next_state = self._extract_state().cpu().numpy()
                reward = self._compute_reward().item()
                done = (step == steps_per_episode - 1)
                
                self.store_experience(state, action, reward, next_state, done)
                self.train_step()
                
                episode_reward += reward
                state = next_state
            
            episode_rewards.append(episode_reward)
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")
            
            if save_path and episode % 10 == 0:
                self.save_model(save_path)
        
        # Plot and save the reward curve
        plt.figure(figsize=(10, 6))
        plt.plot(episode_rewards, label='Episode Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Reward Curve')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plot_path = os.path.join(os.path.dirname(save_path), 'training_rewards.pdf') if save_path else 'training_rewards.pdf'
        plt.savefig(plot_path)
        plt.close()
        print(f"Reward plot saved to {plot_path}")
        
        if save_path:
            self.save_model(save_path)

    def save_model(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])

def main(TRAIN_MODE=False):
    init = Initialization()
    init_spheresheet = Initialization_spheresheet()
    cube_dict = init.create_cells_hex(20, 20, 1, 1.0, 0.5)
    vm = VertexModel(cube_dict, k_bend=2e-18, K_vol=0.25, k_membrane=1e-15, growth_rates=15, time_step=0.01)
    simulator = CoupledSimulator(vm, coupling_strength=0.5, dt=0.01)
    
    #target_vertices = init_spheresheet.create_thick_spherical_shell(outer_radius=10.0, thickness=2.0, device="cuda")
    target_vertices = torch.load("/home/kaiyi/RD_VM/RD_VM_hexagon/2vm_rd_singlelayer/vertices_final_1000.pt")
    MODEL_PATH = "/home/kaiyi/RD_VM/RD_VM_hexagon/shape_predictor_rl_td3_noconstraintaction.pth"
    
    if TRAIN_MODE:
        agent = TD3Agent(simulator, target_vertices)
        agent.train(episodes=1000, steps_per_episode=10, save_path=MODEL_PATH)
    else:
        agent = TD3Agent(simulator, target_vertices, model_path=MODEL_PATH)
    
    optimal_action = agent.choose_action(agent._extract_state().cpu().numpy(), exploration=False)
    print(f"Optimal parameters - Growth rate U: {optimal_action[0]:.4f}, Decay rate V: {optimal_action[1]:.4f}")

if __name__ == "__main__":
    main(TRAIN_MODE=True)