import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)

        # 狀態價值分支
        self.value_stream = nn.Linear(128, 1)
        # 優勢分支
        self.advantage_stream = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # 結合價值和優勢來計算 Q 值
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class DuelingDQNAgent:
    def __init__(self, state_size, action_size, device, learning_rate=0.001, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=64, memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        # 使用 DuelingQNetwork 替代 QNetwork
        self.q_network = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target_network = DuelingQNetwork(state_size, action_size).to(self.device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def update_target_network(self):
        """更新目標網絡的權重"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_action(self, state):
        """基於 ε-greedy 策略選擇動作"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def store_experience(self, state, action, reward, next_state, done):
        """存儲經驗"""
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)
        self.memory.append((state, action, reward, next_state, done))

    def replay_experience(self):
        """經驗回放"""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.stack(states).to(self.device)
        actions = torch.cat(actions).to(self.device)
        rewards = torch.cat(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.cat(dones).to(self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.discount_factor * next_q_values * (1 - dones))

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run_episode(self, env, max_steps):
        """運行一個回合"""
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = self.get_action(state)
            next_state, reward, done, _ = env.step(action)
            self.store_experience(state, action, reward, next_state, done)
            self.replay_experience()

            state = next_state
            total_reward += reward

            if done:
                break

        return total_reward
