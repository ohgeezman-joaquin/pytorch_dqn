import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from method.dqn import DQNAgent
from lunar_lander import LunarLanderEnvironment
import torch
from collections import deque
import os
from method.double_dqn import DoubleDQNAgent
from method.dueling_dqn import DuelingDQNAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(agent, episode, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_episode_{episode}.pth")
    torch.save({
        'episode': episode,
        'model_state_dict': agent.q_network.state_dict(),
        'target_model_state_dict': agent.target_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def main():
    env = LunarLanderEnvironment()
    agent = DuelingDQNAgent(
        state_size=8,
        action_size=4,
        device=device,
        learning_rate=0.0001,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        batch_size=128,
        memory_size=20000
    )

    episodes = 3000
    max_steps = 300
    update_target_network_every = 5
    reward_mean_num = 10

    rewards = []

    
    reward_window = deque(maxlen=reward_mean_num)

    for episode in range(episodes):

        # max_steps = max(1000 - 5 * (episode // 10), 300)
        # max_steps = max(10000 - 300 * (episode // 10), 300)


        total_reward = agent.run_episode(env, max_steps)

        # 每隔 x 個回合更新目標網絡
        if episode % update_target_network_every == 0:
            agent.update_target_network()

        reward_window.append(total_reward)
        rewards.append(total_reward)

        # 每 10 個回合計算並打印平均獎勵
        if episode % 10 == 0:
            average_reward = np.mean(reward_window)
            print(f"Episode: {episode}, Average Reward: {average_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

        # 每 100 個回合顯示 max_steps
        if episode % 100 == 0:
            print("max_steps:", max_steps)

    save_checkpoint(agent, episodes, checkpoint_dir="checkpoints")
    env.close()
    plot_rewards(rewards)

def plot_rewards(rewards):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward Over Episodes")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print(device)
    main()
