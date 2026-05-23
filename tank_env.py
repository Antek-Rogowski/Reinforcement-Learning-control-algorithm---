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
        
        # Inicjalizacja instancji obiektu z Pybind11
        self.sim = tank_sim.ThreeTankSystem(delta_t=0.01)
        
        # Parametry regulacji
        self.target_h3 = 0.5   # Docelowy poziom w zbiorniku 3 (wartość zadana)
        self.max_h = 1.0       # Maksymalna wysokość zbiorników (1 metr)
        self.max_steps = 1000  # Długość jednego epizodu
        self.current_step = 0
        
        # Przestrzeń akcji: ciągły przepływ na wejściu [m^3/s]
        self.action_space = spaces.Box(
            low=0.0, 
            high=0.001, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Przestrzeń obserwacji: poziomy w 3 zbiornikach [m]
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
        
        # Pobieramy stan początkowy z C++
        obs = np.array(self.sim.get_state(), dtype=np.float32)
        return obs, {}

    def step(self, action):
        self.current_step += 1
        
        # Akcja z algorytmu RL
        q_in = float(action[0])
        
        # Serce systemu: wołamy C++, zwalniamy GIL, robimy 10 kroków numerycznych RK4!
        # Czas wykonania ułamki milisekund.
        next_state = self.sim.step(q_in, steps_per_action=10)
        obs = np.array(next_state, dtype=np.float32)
        
        h3 = obs[2]
        
        # Funkcja nagrody w oparciu o uchyb
        error = self.target_h3 - h3
        reward = -float(error ** 2)
        
        # Restrykcyjne warunki końcowe: przelanie zbiornika kończy epizod z karą
        terminated = False
        if np.any(obs > self.max_h):
            reward -= 50.0
            terminated = True
            
        truncated = bool(self.current_step >= self.max_steps)
        
        return obs, reward, terminated, truncated, {}

# --- PĘTLA UCZĄCA ---
if __name__ == "__main__":
    print("Inicjalizacja środowiska...")
    env = ThreeTankEnv()
    
    # Inicjalizacja agenta Proximal Policy Optimization
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tanks_tensorboard/")
    
    print("Rozpoczynamy ciężki trening bez dławienia GILem!")
    # Ten proces pożerałby czas w czystym Pythonie, a tutaj będzie działał niezwykle płynnie.
    model.learn(total_timesteps=100_000)
    
    # Zapis wag po treningu
    model.save("ppo_three_tanks")
    print("Trening zakończony, model zapisany jako ppo_three_tanks.zip.")