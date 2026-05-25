import pygame
import sys
import numpy as np
import tank_sim  # Importujemy Twój ultraszybki silnik C++

# Inicjalizacja silnika graficznego
pygame.init()

# Konfiguracja okna i kolorów
WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Interaktywny Symulator Układu Trzech Zbiorników (C++ & Pygame)")

# Kolory
WHITE = (245, 245, 245)
BLACK = (30, 30, 30)
BLUE = (50, 150, 255)
RED = (220, 50, 50)
GRAY = (180, 180, 180)
DARK_GRAY = (100, 100, 100)

font = pygame.font.SysFont("Arial", 16, bold=True)

# Inicjalizacja fizyki z C++
# dt=0.01s z 10 krokami RK4 na każdą klatkę Pygame daje ładne, płynne tempo symulacji
sim = tank_sim.ThreeTankSystem(delta_t=0.01)

# Parametry rysowania zbiorników
MAX_H = 1.0  # 1.0 m to maksymalna wysokość
TANK_W = 120
TANK_H = 350
tank_x_positions = [150, 400, 650]
base_y = 450  # Współrzędna Y dna zbiorników

# Wartość zadana
TARGET_H = 0.25

# Parametry suwaka sterowania zaworem
slider_rect = pygame.Rect(150, 520, 600, 20)
slider_knob_rect = pygame.Rect(150, 510, 20, 40)
dragging_slider = False
q_in_max = 0.001
current_q_in = 0.0

clock = pygame.time.Clock()

def draw_tank(x, y_bottom, width, height_px, water_level_m):
    # Rysowanie tła zbiornika
    rect_bg = pygame.Rect(x, y_bottom - height_px, width, height_px)
    pygame.draw.rect(screen, GRAY, rect_bg, border_radius=5)
    
    # Rysowanie wody
    water_h_px = int((water_level_m / MAX_H) * height_px)
    # Zabezpieczenie przed rysowaniem poza zbiornikiem
    water_h_px = min(water_h_px, height_px) 
    
    if water_h_px > 0:
        rect_water = pygame.Rect(x, y_bottom - water_h_px, width, water_h_px)
        pygame.draw.rect(screen, BLUE, rect_water, border_radius=5)
        
    # Rysowanie obramowania
    pygame.draw.rect(screen, DARK_GRAY, rect_bg, width=3, border_radius=5)
    
    # Podpis poziomu wody
    text = font.render(f"{water_level_m:.3f} m", True, BLACK)
    screen.blit(text, (x + width//2 - text.get_width()//2, y_bottom + 10))

running = True
while running:
    # 1. OBSŁUGA ZDARZEŃ
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        # Obsługa kliknięcia i przeciągania suwaka
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and slider_rect.collidepoint(event.pos):
                dragging_slider = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                dragging_slider = False
        elif event.type == pygame.MOUSEMOTION:
            if dragging_slider:
                mouse_x = event.pos[0]
                # Ograniczenie suwaka do paska
                new_x = max(slider_rect.left, min(mouse_x, slider_rect.right))
                slider_knob_rect.centerx = new_x
                # Przeliczenie pozycji suwaka na wartość przepływu (0.0 do 0.001)
                percent = (new_x - slider_rect.left) / slider_rect.width
                current_q_in = percent * q_in_max

    # 2. OBLICZENIA FIZYKI (Natywny moduł C++)
    # Wykonujemy kroki symulacji dla aktualnie ustawionego przepływu
    state = sim.step(current_q_in, steps_per_action=10)
    h1, h2, h3 = state[0], state[1], state[2]

    # 3. RENDEROWANIE GRAFIKI
    screen.fill(WHITE)
    
    # Rysowanie rur łączących (estetyka)
    pygame.draw.line(screen, DARK_GRAY, (tank_x_positions[0]+TANK_W, base_y-20), (tank_x_positions[1], base_y-20), 8)
    pygame.draw.line(screen, DARK_GRAY, (tank_x_positions[1]+TANK_W, base_y-20), (tank_x_positions[2], base_y-20), 8)
    pygame.draw.line(screen, DARK_GRAY, (tank_x_positions[2]+TANK_W, base_y-20), (tank_x_positions[2]+TANK_W+30, base_y-20), 8)

    # Rysowanie zbiorników
    draw_tank(tank_x_positions[0], base_y, TANK_W, TANK_H, h1)
    draw_tank(tank_x_positions[1], base_y, TANK_W, TANK_H, h2)
    draw_tank(tank_x_positions[2], base_y, TANK_W, TANK_H, h3)

    # Linia wartości zadanej na 3 zbiorniku
    target_y = base_y - int((TARGET_H / MAX_H) * TANK_H)
    pygame.draw.line(screen, RED, (tank_x_positions[2] - 10, target_y), (tank_x_positions[2] + TANK_W + 10, target_y), 3)
    target_text = font.render(f"Cel: {TARGET_H} m", True, RED)
    screen.blit(target_text, (tank_x_positions[2] + TANK_W + 15, target_y - 10))

    # Rysowanie UI - Suwak
    pygame.draw.rect(screen, DARK_GRAY, slider_rect, border_radius=10)
    pygame.draw.rect(screen, BLACK, slider_knob_rect, border_radius=5)
    
    # Tekst suwaka
    ctrl_text = font.render(f"Otwarcie zaworu (q_in): {current_q_in:.5f} m^3/s", True, BLACK)
    screen.blit(ctrl_text, (WIDTH//2 - ctrl_text.get_width()//2, 480))

    pygame.display.flip()
    
    # Utrzymanie stabilnych 60 klatek na sekundę
    clock.tick(60)

pygame.quit()
sys.exit()