import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

# Importujemy nasz błyskawiczny moduł skompilowany w C++!
import tank_sim 

class ThreeTankEnv(gym.Env):
    """
    Środowisko RL opakowujące symulator C++ układu trzech połączonych zbiorników.
    """
    def __init__(self):
        super().__init__()
        
        self.sim = tank_sim.ThreeTankSystem(delta_t=0.01)
        
        self.target_h3 = 0.25
        self.prev_action = 0.0
        self.max_h = 1.0      
        self.max_steps = 4000 
        self.current_step = 0
        
        self.action_space = spaces.Box(
            low=0.0, 
            high=0.001, 
            shape=(1,), 
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=0.0, 
            high=self.max_h, 
            shape=(3,), 
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.current_step = 0
        
        obs = np.array(self.sim.get_state(), dtype=np.float32)
        return obs, {}

    def step(self, action):
        self.current_step += 1
        delta_action = q_in - self.prev_action
        self.prev_action = q_in
        
        q_in = float(action[0])
        
        next_state = self.sim.step(q_in, steps_per_action=10)
        obs = np.array(next_state, dtype=np.float32)
        
        h3 = obs[2]
        
        error = self.target_h3 - h3
        reward = -float(error ** 2) - 0.5 * float(delta_action ** 2)
        
        terminated = False
        if np.any(obs > self.max_h):
            reward -= 50.0
            terminated = True
            
        truncated = bool(self.current_step >= self.max_steps)
        
        return obs, reward, terminated, truncated, {}


if __name__ == "__main__":
    print("Inicjalizacja środowiska...")
    env = ThreeTankEnv()
    
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tanks_tensorboard/")
    
    print("Rozpoczynamy ciężki trening bez dławienia GILem!")
    model.learn(total_timesteps=1_000_000)
    
    model.save("ppo_three_tanks")
    print("Trening zakończony, model zapisany jako ppo_three_tanks.zip.")