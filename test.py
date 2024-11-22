import torch
import time
import numpy as np
from dqn import DQNAgent
from lunar_lander import LunarLanderEnvironment
import gymnasium as gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "checkpoints/checkpoint_episode_2999.pth"

def load_checkpoint(agent, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    agent.q_network.load_state_dict(checkpoint['model_state_dict'])
    agent.target_network.load_state_dict(checkpoint['target_model_state_dict'])
    print(f"Checkpoint loaded from {checkpoint_path}")

def run_inference(agent, env, episodes=100, max_steps=300, render_delay=5):
    """
    執行模型推理並顯示 LunarLander 動畫

    參數:
    - agent: 訓練好的 DQNAgent
    - env: 環境 (LunarLander)
    - episodes: 要執行的回合數
    - max_steps: 每回合的最大步數
    - render_delay: 動畫顯示的間隔時間 (秒)
    """
    for episode in range(episodes):
        state, _ = env.reset()  # 確保 state 正確初始化
        total_reward = 0
        done = False

        for step in range(max_steps):
            # 確保 state 是 tensor 並加上 batch 維度
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            
            # 使用模型選擇動作
            action = agent.get_action(state_tensor)
            next_state, reward, done, _, _ = env.step(action)

            # 累加回合中的總獎勵
            total_reward += reward

            # 更新狀態
            state = next_state

            # 顯示動畫
            # env.render()
            # time.sleep(render_delay)  # 控制每幀顯示的間隔時間

            # 若遊戲結束 (例如成功著陸或失敗墜毀)，則跳出回合
            if done:
                break
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()

def main():
    # 建立環境與 DQN agent
    env = gym.make("LunarLander-v3", render_mode="human")
    # env.metadata["render_fps"] = 500
    agent = DQNAgent(
        state_size=8,
        action_size=4,
        device=device,
        learning_rate=0.0001,
        discount_factor=0.99,
        epsilon=0.01,  # 減少隨機性，使用模型進行推理
        epsilon_decay=1.0,  # 固定 epsilon，不再衰減
        epsilon_min=0.01,
        batch_size=128,
        memory_size=20000
    )

    # 加載 checkpoint
    load_checkpoint(agent, checkpoint_path)

    # 運行推理並顯示動畫
    run_inference(agent, env, render_delay=0.05)

if __name__ == "__main__":
    main()
