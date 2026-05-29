import pygame
import sys
import numpy as np
import tank_sim
from stable_baselines3 import PPO

pygame.init()
WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("APC Digital Twin: Manual vs AI Control")

WHITE, BLACK, BLUE, RED, GRAY, DARK_GRAY, GREEN = (245, 245, 245), (30, 30, 30), (50, 150, 255), (220, 50, 50), (180, 180, 180), (100, 100, 100), (40, 200, 70)
font = pygame.font.SysFont("Arial", 16, bold=True)
big_font = pygame.font.SysFont("Arial", 24, bold=True)

sim = tank_sim.ThreeTankSystem(delta_t=0.01)

try:
    model = PPO.load("ppo_three_tanks")
except Exception as e:
    print("Brak modelu. Najpierw uruchom tank_env.py")
    sys.exit()

MAX_H, TARGET_H = 1.0, 0.25
TANK_W, TANK_H = 120, 350
tank_x_positions = [150, 400, 650]
base_y = 450

slider_rect = pygame.Rect(150, 520, 600, 20)
slider_knob_rect = pygame.Rect(150, 510, 20, 40)
dragging_slider = False
q_in_max = 0.001
current_q_in = 0.0
integral_error = 0.0
ai_mode = False  

clock = pygame.time.Clock()

def draw_tank(x, y_bottom, width, height_px, water_level_m):
    rect_bg = pygame.Rect(x, y_bottom - height_px, width, height_px)
    pygame.draw.rect(screen, GRAY, rect_bg, border_radius=5)
    water_h_px = int((water_level_m / MAX_H) * height_px)
    water_h_px = min(water_h_px, height_px) 
    
    if water_h_px > 0:
        color = RED if water_level_m >= 0.999 else BLUE
        rect_water = pygame.Rect(x, y_bottom - water_h_px, width, water_h_px)
        pygame.draw.rect(screen, color, rect_water, border_radius=5)
        
    pygame.draw.rect(screen, DARK_GRAY, rect_bg, width=3, border_radius=5)
    text = font.render(f"{water_level_m:.3f} m", True, BLACK)
    screen.blit(text, (x + width//2 - text.get_width()//2, y_bottom + 10))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            ai_mode = not ai_mode
        elif event.type == pygame.MOUSEBUTTONDOWN and not ai_mode:
            if event.button == 1 and slider_rect.collidepoint(event.pos): dragging_slider = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            dragging_slider = False
        elif event.type == pygame.MOUSEMOTION and not ai_mode:
            if dragging_slider:
                new_x = max(slider_rect.left, min(event.pos[0], slider_rect.right))
                current_q_in = ((new_x - slider_rect.left) / slider_rect.width) * q_in_max

    current_state = sim.get_state()
    h1, h2, h3 = current_state[0], current_state[1], current_state[2]
    
    error = TARGET_H - h3
    integral_error = max(-10.0, min(10.0, integral_error + error * 0.1))
    current_action_norm = current_q_in / q_in_max

    if ai_mode:
        obs = np.array([h1, h2, h3, current_action_norm, integral_error], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        action_clamped = max(-1.0, min(float(action[0]), 1.0))
        current_action_norm = (action_clamped + 1.0) / 2.0
        current_q_in = current_action_norm * q_in_max

    slider_knob_rect.centerx = slider_rect.left + int(current_action_norm * slider_rect.width)
    state = sim.step(current_q_in, steps_per_action=10)

    screen.fill(WHITE)
    pygame.draw.line(screen, DARK_GRAY, (tank_x_positions[0]+TANK_W, base_y-20), (tank_x_positions[1], base_y-20), 8)
    pygame.draw.line(screen, DARK_GRAY, (tank_x_positions[1]+TANK_W, base_y-20), (tank_x_positions[2], base_y-20), 8)
    pygame.draw.line(screen, DARK_GRAY, (tank_x_positions[2]+TANK_W, base_y-20), (tank_x_positions[2]+TANK_W+30, base_y-20), 8)

    draw_tank(tank_x_positions[0], base_y, TANK_W, TANK_H, state[0])
    draw_tank(tank_x_positions[1], base_y, TANK_W, TANK_H, state[1])
    draw_tank(tank_x_positions[2], base_y, TANK_W, TANK_H, state[2])

    target_y = base_y - int((TARGET_H / MAX_H) * TANK_H)
    pygame.draw.line(screen, RED, (tank_x_positions[2] - 10, target_y), (tank_x_positions[2] + TANK_W + 10, target_y), 3)
    
    pygame.draw.rect(screen, DARK_GRAY, slider_rect, border_radius=10)
    pygame.draw.rect(screen, GREEN if ai_mode else BLACK, slider_knob_rect, border_radius=5)
    
    screen.blit(big_font.render("STEROWANIE: AI (PPO)" if ai_mode else "STEROWANIE: MANUALNE (SPACJA)", True, GREEN if ai_mode else BLACK), (WIDTH//2 - 150, 30))
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()