import gymnasium as gym

class LunarLanderEnvironment:
    def __init__(self):
        """
        初始化 Lunar Lander 環境
        """
        self.env = gym.make('LunarLander-v3')
        
    def reset(self):
        """
        重置環境
        
        Returns:
            np.array: 初始狀態
        """
        return self.env.reset()[0]
    
    def step(self, action):
        """
        執行一個動作
        
        Args:
            action (int): 要執行的動作
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        next_state, reward, done, truncated, info = self.env.step(action)
        return next_state, reward, done, info
    
    def render(self):
        """
        渲染環境
        """
        self.env.render()
    
    def close(self):
        """
        關閉環境
        """
        self.env.close()