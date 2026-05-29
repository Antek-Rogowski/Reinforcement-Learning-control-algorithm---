import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

# Importujemy nasz błyskawiczny moduł skompilowany w C++
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
        
        # Symetryczna przestrzeń akcji znormalizowana dla PPO [-1.0, 1.0]
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Obserwacja rozszerzona o target_h3: [h1, h2, h3, target]
        self.observation_space = spaces.Box(
            low=0.0, 
            high=self.max_h, 
            shape=(5,), 
            dtype=np.float32
        )

    def _get_obs(self):
        """Metoda pomocnicza tworząca wektor obserwacji."""
        state = self.sim.get_state()
        return np.array([state[0], state[1], state[2], self.target_h3], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.current_step = 0
        self.prev_action = 0.0
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # Przeskalowanie decyzji modelu ([-1, 1]) na przepływ zaworu ([0.0, 0.001])
        q_in = 0.0005 * (float(action[0]) + 1.0)
        
        # Wyliczenie różnicy stymulacji względem poprzedniego kroku (naprawiony błąd zmiennej)
        delta_action = q_in - self.prev_action
        self.prev_action = q_in
        
        # Krok symulacji w C++
        self.sim.step(q_in, steps_per_action=10)
        
        # Pobranie stanu
        obs = self._get_obs()
        h3 = obs[2]
        
        # Obliczenie nagrody. Normalizujemy delte akcji, aby nie zniknęła przy liczeniu małych kwadratów
        error = self.target_h3 - h3
        reward = -float(error ** 2) - 0.5 * float((delta_action / 0.001) ** 2)
        
        # Usunięto wczesną terminację po przelaniu. Pozwalamy fizyce na działanie (patrz tanks.cpp), 
        # a model karany jest po prostu za nieutrzymywanie zadanego punktu.
        terminated = False
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