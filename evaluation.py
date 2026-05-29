import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from tank_env import ThreeTankEnv

def evaluate_and_plot():
    print("Inicjalizacja środowiska ewaluacyjnego...")
    env = ThreeTankEnv()
    
    print("Ładowanie modelu...")
    model = PPO.load("ppo_three_tanks")
    
    h1_hist = []
    h2_hist = []
    h3_hist = []
    action_hist = []
    
    obs, info = env.reset()
    done = False
    
    print("Symulacja obiektu...")
    while not done:
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        
        h1_hist.append(obs[0])
        h2_hist.append(obs[1])
        h3_hist.append(obs[2])
        
        # Przeliczamy znormalizowaną akcję z powrotem na fizyczny przepływ na potrzeby wykresu
        q_in = 0.0005 * (float(action[0]) + 1.0)
        action_hist.append(q_in)
        
        done = terminated or truncated

    time_axis = np.arange(len(h1_hist)) * 0.1

    print("Generowanie wykresów...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(time_axis, h1_hist, label='Zbiornik 1 (h1)', linestyle='--')
    ax1.plot(time_axis, h2_hist, label='Zbiornik 2 (h2)', linestyle='-.')
    ax1.plot(time_axis, h3_hist, label='Zbiornik 3 (h3) - Wyjście', linewidth=2)
    
    target = env.target_h3
    ax1.axhline(y=target, color='r', linestyle=':', label=f'Wartość zadana ({target} m)')
    
    ax1.set_ylabel('Poziom cieczy [m]')
    ax1.set_title('Odpowiedź układu trzech zbiorników ze sterowaniem RL (PPO)')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    ax2.step(time_axis, action_hist, label='Przepływ wejściowy', color='purple')
    ax2.set_xlabel('Czas symulacji [s]')
    ax2.set_ylabel('Przepływ [m³/s]')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_and_plot()