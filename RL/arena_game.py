import pygame
import random
import math

class ArenaGame:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("BCI Adaptive Game - Neon Arena")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Player Stats
        self.player_pos = [width // 2, height // 2]
        self.player_speed = 5.0
        self.player_radius = 15
        
        # Firing Logic
        self.projectiles = []
        self.fire_cooldown = 0
        self.fire_rate = 10  # frames between shots
        self.bullet_speed = 12.0
        
        # Enemies
        self.enemies = []
        self.base_spawn_frames = 60
        
        # Particles
        self.particles = []
        
        # Physics Engine
        self.score = 0
        self.frame_count = 0
        self.recent_collisions = 0
        
        # BCI Variables
        self.speed_multiplier = 1.0
        self.spawn_rate_multiplier = 1.0
        self.base_enemy_speed = 2.0
        
        self.current_workload = 0.6
        self.smooth_workload = 0.6
        self.current_difficulty = 0.5
        self.smooth_difficulty = 0.5
        self.action_label = "Maintain"
        self.flow_zone = True

        pygame.font.init()
        self.font = pygame.font.SysFont("Courier", 18, bold=True)
        
        # Screen Shake
        self.shake_frames = 0
        
    def update_bci_state(self, workload, difficulty, action_label, flow_zone, game_params=None):
        self.current_workload = workload
        self.current_difficulty = difficulty
        self.action_label = action_label
        self.flow_zone = flow_zone
        
        if game_params:
            self.speed_multiplier = game_params.get('enemy_speed', 1.0)
            self.spawn_rate_multiplier = game_params.get('spawn_rate', 1.0)
            
    def update_workload_live(self, live_workload):
        self.current_workload = live_workload

    def spawn_particle_explosion(self, x, y, color, count=15):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2.0, 6.0)
            life = random.randint(20, 40)
            self.particles.append([
                [x, y],                      # position
                [math.cos(angle)*speed, math.sin(angle)*speed],  # velocity
                color,                       # color
                life,                        # life
                life                         # max life
            ])

    def process_frame(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        # Fade trail effect (Additive motion blur aesthetic)
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(80)
        overlay.fill((5, 5, 10))
        self.screen.blit(overlay, (0, 0))
        
        self.smooth_workload += (self.current_workload - self.smooth_workload) * 0.1
        self.smooth_difficulty += (self.current_difficulty - self.smooth_difficulty) * 0.05
        
        self.update_mechanics()
        self.draw()
        
        self.clock.tick(60)
        self.frame_count += 1
        return self.running

    def update_mechanics(self):
        # 1. Player WASD Movement
        keys = pygame.key.get_pressed()
        dx, dy = 0, 0
        if keys[pygame.K_a]: dx -= 1
        if keys[pygame.K_d]: dx += 1
        if keys[pygame.K_w]: dy -= 1
        if keys[pygame.K_s]: dy += 1
        
        # Normalize diagonal speed
        if dx != 0 and dy != 0:
            length = math.hypot(dx, dy)
            dx /= length
            dy /= length
            
        self.player_pos[0] += dx * self.player_speed
        self.player_pos[1] += dy * self.player_speed
        
        # Clamp to bounds
        self.player_pos[0] = max(20, min(self.width - 20, self.player_pos[0]))
        self.player_pos[1] = max(100, min(self.height - 20, self.player_pos[1])) # Keep away from HUD

        # 2. Player Arrow Key Firing
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
            
        shoot_dx, shoot_dy = 0, 0
        if keys[pygame.K_LEFT]: shoot_dx -= 1
        elif keys[pygame.K_RIGHT]: shoot_dx += 1
        if keys[pygame.K_UP]: shoot_dy -= 1
        elif keys[pygame.K_DOWN]: shoot_dy += 1
        
        if (shoot_dx != 0 or shoot_dy != 0) and self.fire_cooldown <= 0:
            # Normalize fire vector
            length = math.hypot(shoot_dx, shoot_dy)
            shoot_dx /= length
            shoot_dy /= length
            self.projectiles.append([
                list(self.player_pos),
                [shoot_dx * self.bullet_speed, shoot_dy * self.bullet_speed]
            ])
            self.fire_cooldown = self.fire_rate

        # 3. Update Projectiles
        for proj in self.projectiles[:]:
            proj[0][0] += proj[1][0]
            proj[0][1] += proj[1][1]
            if not (-20 < proj[0][0] < self.width+20 and -20 < proj[0][1] < self.height+20):
                self.projectiles.remove(proj)

        # 4. Spawn Enemies
        dynamic_hz = max(5, int(self.base_spawn_frames / max(0.2, self.spawn_rate_multiplier)))
        if self.frame_count % dynamic_hz == 0:
            # Spawn along borders
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top': spawn = [random.randint(0, self.width), 100] # Below HUD
            elif edge == 'bottom': spawn = [random.randint(0, self.width), self.height]
            elif edge == 'left': spawn = [0, random.randint(100, self.height)]
            else: spawn = [self.width, random.randint(100, self.height)]
            
            # HP, Size, Speed
            self.enemies.append([spawn, 10, self.base_enemy_speed * self.speed_multiplier])

        # 5. Update Enemies & Collision
        for enemy in self.enemies[:]:
            # Seek Player
            ex, ey = enemy[0]
            px, py = self.player_pos
            angle = math.atan2(py - ey, px - ex)
            enemy[0][0] += math.cos(angle) * enemy[2]
            enemy[0][1] += math.sin(angle) * enemy[2]
            
            # Check Collision with Player (Trauma event)
            if math.hypot(px - ex, py - ey) < (self.player_radius + enemy[1]):
                self.score -= 50
                self.recent_collisions += 1 # PIPES TO SIMULATOR
                self.shake_frames = 10
                self.spawn_particle_explosion(ex, ey, (255, 50, 50), 30)
                self.enemies.remove(enemy)
                continue
                
            # Check Collision with Projectiles
            hit = False
            for proj in self.projectiles[:]:
                p_x, p_y = proj[0]
                if math.hypot(p_x - ex, p_y - ey) < (enemy[1] + 5):
                    self.score += 10
                    self.spawn_particle_explosion(ex, ey, (255, 100, 200), random.randint(8, 15))
                    self.projectiles.remove(proj)
                    hit = True
                    break
            
            if hit:
                self.enemies.remove(enemy)

        # 6. Update Particles
        for part in self.particles[:]:
            part[0][0] += part[1][0]
            part[0][1] += part[1][1]
            part[1][0] *= 0.9  # friction
            part[1][1] *= 0.9
            part[3] -= 1       # decay life
            if part[3] <= 0:
                self.particles.remove(part)

        if self.shake_frames > 0:
            self.shake_frames -= 1

    def draw_hud(self):
        s = pygame.Surface((self.width, 80))
        s.set_alpha(220)
        s.fill((15, 15, 20))
        self.screen.blit(s, (0, 0))
        
        # Bottom neon border for HUD
        pygame.draw.line(self.screen, (0, 255, 255), (0, 80), (self.width, 80), 2)

        # Workload Gauge (Center)
        wl_text = self.font.render(f"W: {self.smooth_workload:.2f}", True, (200, 200, 200))
        self.screen.blit(wl_text, (self.width//2 - 150, 10))
        pygame.draw.rect(self.screen, (40, 40, 50), (self.width//2 - 150, 40, 300, 15), border_radius=4)
        gap = abs(self.smooth_workload - 0.6)
        wl_color = (0, 255, 255) if gap < 0.10 else ((255, 50, 100) if self.smooth_workload > 0.6 else (100, 100, 255))
        pygame.draw.rect(self.screen, wl_color, (self.width//2 - 150, 40, int(300 * max(0, min(1, self.smooth_workload))), 15), border_radius=4)
        
        # Target Zone tick
        target_x = (self.width//2 - 150) + int(300 * 0.6)
        pygame.draw.line(self.screen, (255, 255, 255), (target_x, 35), (target_x, 60), 2)
        
        # Difficulty Gauge (Left)
        diff_text = self.font.render(f"D: {self.smooth_difficulty:.2f}", True, (200, 200, 200))
        self.screen.blit(diff_text, (20, 10))
        pygame.draw.rect(self.screen, (40, 40, 50), (20, 40, 150, 10), border_radius=3)
        pygame.draw.rect(self.screen, (255, 100, 0), (20, 40, int(150 * max(0, min(1, self.smooth_difficulty))), 10), border_radius=3)
        
        # Action Log & Score (Right)
        act_color = (0, 255, 255) if self.action_label == "Maintain" else (255, 100, 200)
        act_text = self.font.render(f"ACTION: {self.action_label}", True, act_color)
        self.screen.blit(act_text, (self.width - 220, 20))
        
        score_t = self.font.render(f"SCORE: {self.score}", True, (255, 255, 0))
        self.screen.blit(score_t, (self.width - 220, 45))

    def draw(self):
        shake_x, shake_y = 0, 0
        if self.shake_frames > 0:
            shake_x = random.randint(-4, 4)
            shake_y = random.randint(-4, 4)
            
        # Draw Background Grid
        for x in range(0, self.width, 50):
            pygame.draw.line(self.screen, (25, 25, 35), (x + shake_x, 80), (x + shake_x, self.height), 1)
        for y in range(80, self.height, 50):
            pygame.draw.line(self.screen, (25, 25, 35), (shake_x, y + shake_y), (self.width, y + shake_y), 1)

        # Draw Particles
        for part in self.particles:
            alpha = max(0, int(255 * (part[3] / part[4])))
            px, py = int(part[0][0] + shake_x), int(part[0][1] + shake_y)
            # Additive glow trick
            pygame.draw.circle(self.screen, part[2], (px, py), 3)

        # Draw Projectiles
        for proj in self.projectiles:
            px, py = int(proj[0][0] + shake_x), int(proj[0][1] + shake_y)
            pygame.draw.circle(self.screen, (0, 255, 255), (px, py), 4)

        # Draw Enemies
        for enemy in self.enemies:
            ex, ey = int(enemy[0][0] + shake_x), int(enemy[0][1] + shake_y)
            r = enemy[1]
            pygame.draw.polygon(self.screen, (255, 0, 50), [
                (ex, ey - r),
                (ex - r, ey + r),
                (ex + r, ey + r)
            ], 2)
            # Glow core
            pygame.draw.circle(self.screen, (255, 100, 100), (ex, ey + r//3), 2)

        # Draw Player
        px, py = int(self.player_pos[0] + shake_x), int(self.player_pos[1] + shake_y)
        pygame.draw.circle(self.screen, (0, 255, 255), (px, py), self.player_radius, 2)
        # Core
        pygame.draw.circle(self.screen, (255, 255, 255), (px, py), 5)

        self.draw_hud()
        pygame.display.flip()

    def quit(self):
        pygame.quit()

# Standardized entry point
AdaptiveGame = ArenaGame
