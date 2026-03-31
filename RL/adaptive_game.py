import pygame
import random
import math

class AscensionGame:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("BCI Adaptive Game - Neon Ascension")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Player Physics Stats
        self.player_x = 150
        self.player_y = height // 2
        self.player_dy = 0.0          # Vertical Velocity
        self.gravity = 0.45           # Constant downward acceleration
        self.thrust = -8.0            # Instant upward impulse when SPACE is pressed
        self.player_radius = 16
        
        # Obstacles (Gates)
        # Each gate is [x_position, gap_center_y]
        self.gates = []
        self.gate_width = 80
        
        # Base Simulation Variables (scaled by Difficulty)
        self.base_scroll_speed = 4.0
        self.base_gap_height = 250
        self.base_spawn_distance = 400  # Pixels between gates
        
        # Game/BCI metrics
        self.distance_scrolled = 0.0
        self.last_spawn_x = 0.0
        
        # Physics Engine Feedback
        self.score = 0
        self.frame_count = 0
        self.recent_collisions = 0
        self.shake_frames = 0
        self.particles = []
        
        # BCI State Interpolation
        self.speed_multiplier = 1.0
        self.spawn_rate_multiplier = 1.0 # not heavily used here, speed handles it
        
        self.current_workload = 0.6
        self.smooth_workload = 0.6
        self.current_difficulty = 0.5
        self.smooth_difficulty = 0.5
        self.action_label = "Maintain"
        self.flow_zone = True

        pygame.font.init()
        self.font = pygame.font.SysFont("Courier", 18, bold=True)
        
    def update_bci_state(self, workload, difficulty, action_label, flow_zone, game_params=None):
        self.current_workload = workload
        self.current_difficulty = difficulty
        self.action_label = action_label
        self.flow_zone = flow_zone
        
        if game_params:
            self.speed_multiplier = game_params.get('enemy_speed', 1.0)
            
    def update_workload_live(self, live_workload):
        self.current_workload = live_workload

    def spawn_particle_explosion(self, x, y, color, count=15):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2.0, 8.0)
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
            # Spacebar Thruster Event
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.player_dy = self.thrust
                    # Small exhaust particle puff
                    self.spawn_particle_explosion(self.player_x, self.player_y + 10, (100, 255, 255), 5)

        # Draw a beautiful dark fading background
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(100) # Creates organic motion blur for player
        overlay.fill((5, 10, 20))
        self.screen.blit(overlay, (0, 0))
        
        self.smooth_workload += (self.current_workload - self.smooth_workload) * 0.1
        self.smooth_difficulty += (self.current_difficulty - self.smooth_difficulty) * 0.05
        
        self.update_mechanics()
        self.draw()
        
        self.clock.tick(60)
        self.frame_count += 1
        return self.running

    def update_mechanics(self):
        # 1. Player Physics (Gravity)
        self.player_dy += self.gravity
        self.player_y += self.player_dy
        
        # Limit terminal velocity so it doesn't glitch through objects
        if self.player_dy > 12.0:
            self.player_dy = 12.0
            
        # 2. Floor / Ceiling Collision Crash
        # The HUD is 80px tall, so play area is y=80 to y=height
        if self.player_y < 80 + self.player_radius:
            self.player_y = 80 + self.player_radius
            self.player_dy = 2.0 # bounce down
            self.handle_crash()
            
        elif self.player_y > self.height - self.player_radius:
            self.player_y = self.height - self.player_radius
            self.player_dy = -4.0 # bounce up
            self.handle_crash()

        # 3. Dynamic Calculus (BCI Hook)
        # As smooth_difficulty -> 1.0 (Hard), speed increases 3X, and gaps shrink massively!
        current_speed = self.base_scroll_speed * (0.5 + self.smooth_difficulty * 2.5)
        # Easy (0.1) -> Gap 250px. Hard (0.9) -> Gap 90px (very tight!)
        current_gap_height = max(70, self.base_gap_height - (self.smooth_difficulty * 180))
        
        self.distance_scrolled += current_speed

        # 4. Procedural Gate Spawning
        spawn_threshold = self.last_spawn_x + self.base_spawn_distance
        if self.distance_scrolled > spawn_threshold:
            self.last_spawn_x = self.distance_scrolled
            
            # Keep the gap center safely away from ceiling/floor bounds
            min_y = 80 + int(current_gap_height * 0.5) + 30
            max_y = self.height - int(current_gap_height * 0.5) - 30
            gap_center = random.randint(min_y, max_y)
            
            # Append new gate [x_coord, gap_center, gap_height_at_spawn, scored_flag]
            self.gates.append([self.width + 50, gap_center, current_gap_height, False])

        # 5. Move Gates & Check Gate Collisions
        player_rect = pygame.Rect(self.player_x - self.player_radius, self.player_y - self.player_radius, 
                                  self.player_radius*2, self.player_radius*2)
                                  
        for gate in self.gates[:]:
            gate[0] -= current_speed
            
            gx, g_center, g_height, scored = gate
            
            # Top Pipe Bounding Box
            top_pipe = pygame.Rect(gx, 80, self.gate_width, int(g_center - 80 - (g_height/2)))
            # Bottom Pipe Bounding Box
            bottom_pipe_y = int(g_center + (g_height/2))
            bottom_pipe = pygame.Rect(gx, bottom_pipe_y, self.gate_width, self.height - bottom_pipe_y)
            
            # Check Collision
            if top_pipe.colliderect(player_rect) or bottom_pipe.colliderect(player_rect):
                self.handle_crash()
                # Push player back slightly to prevent sticking
                self.player_x -= current_speed 
                if self.player_x < 50:
                    self.player_x = 50 # don't fall off screen entirely
            
            # Check Score (Successfully passed the gate!)
            if not scored and self.player_x > (gx + self.gate_width):
                gate[3] = True
                self.score += 10
                # Give the player a little 'success' particle pop
                self.spawn_particle_explosion(self.player_x, self.player_y, (0, 255, 0), 5)
                
            # Slowly drag player back to center x=150 if they got pushed back
            if self.player_x < 150 and self.frame_count % 3 == 0:
                self.player_x += 1
                
            # Cleanup off-screen gates
            if gate[0] < -self.gate_width:
                self.gates.remove(gate)

        # 6. Update Physics Particles
        for part in self.particles[:]:
            part[0][0] += part[1][0]
            part[0][1] += part[1][1]
            part[1][0] *= 0.95  # friction
            part[1][1] += 0.2   # gravity on sparks!
            part[3] -= 1        # decay life
            if part[3] <= 0:
                self.particles.remove(part)

        if self.shake_frames > 0:
            self.shake_frames -= 1
            
    def handle_crash(self):
        """The true cognitive trigger to simulate failure on Hard Mode"""
        # Only register a new collision pulse every 15 frames max so we don't spam 60 spikes per second
        if self.frame_count % 15 == 0:
            self.score -= 20
            self.recent_collisions += 1 # Sends direct panic to AI Workload Simulator
            self.shake_frames = 12
            self.spawn_particle_explosion(self.player_x + self.player_radius, self.player_y, (255, 50, 50), 20)

    def draw_hud(self):
        # Top HUD Bar
        pygame.draw.rect(self.screen, (10, 15, 25), (0, 0, self.width, 80))
        pygame.draw.line(self.screen, (255, 50, 255), (0, 80), (self.width, 80), 3)

        # Workload Gauge (Center)
        wl_text = self.font.render(f"W: {self.smooth_workload:.2f}", True, (200, 200, 200))
        self.screen.blit(wl_text, (self.width//2 - 150, 10))
        pygame.draw.rect(self.screen, (30, 30, 40), (self.width//2 - 150, 40, 300, 15), border_radius=4)
        
        # Color workload dynamically
        gap = abs(self.smooth_workload - 0.6)
        wl_color = (0, 255, 255) if gap < 0.10 else ((255, 50, 100) if self.smooth_workload > 0.6 else (100, 100, 255))
        pygame.draw.rect(self.screen, wl_color, (self.width//2 - 150, 40, int(300 * max(0, min(1, self.smooth_workload))), 15), border_radius=4)
        
        # Flow Zone 0.6 Tick
        target_x = (self.width//2 - 150) + int(300 * 0.6)
        pygame.draw.line(self.screen, (255, 255, 255), (target_x, 35), (target_x, 60), 2)
        
        # Difficulty Gauge (Left)
        diff_text = self.font.render(f"D: {self.smooth_difficulty:.2f}", True, (200, 200, 200))
        self.screen.blit(diff_text, (20, 10))
        pygame.draw.rect(self.screen, (30, 30, 40), (20, 40, 150, 10), border_radius=3)
        pygame.draw.rect(self.screen, (255, 100, 0), (20, 40, int(150 * max(0, min(1, self.smooth_difficulty))), 10), border_radius=3)
        
        # Action Log & Score (Right)
        act_color = (0, 255, 255) if self.action_label == "Maintain" else ((255, 50, 100) if self.action_label == "Decrease" else (100, 255, 100))
        act_text = self.font.render(f"ACTION: {self.action_label}", True, act_color)
        self.screen.blit(act_text, (self.width - 220, 20))
        
        score_t = self.font.render(f"SCORE: {self.score}", True, (255, 255, 0))
        self.screen.blit(score_t, (self.width - 220, 45))

    def draw(self):
        shake_x, shake_y = 0, 0
        if self.shake_frames > 0:
            shake_x = random.randint(-6, 6)
            shake_y = random.randint(-6, 6)
            
        # Draw Background Scrolling Grid (Parallax impression)
        grid_offset = int(self.distance_scrolled * 0.5) % 100
        for x in range(0 - grid_offset, self.width, 100):
            pygame.draw.line(self.screen, (20, 30, 45), (x + shake_x, 80), (x + shake_x, self.height), 1)

        # Draw Gates (Pipes)
        for gx, g_center, g_height, scored in self.gates:
            gx_shake = int(gx + shake_x)
            
            # Colors scale based on how tight the gap is! (Redder if hard)
            gate_color = (255, 50, 255) if g_height > 150 else (255, 50, 50)
            
            # Top Pipe
            top_rect = (gx_shake, 80, self.gate_width, int(g_center - 80 - (g_height/2)))
            pygame.draw.rect(self.screen, gate_color, top_rect, 0, border_bottom_left_radius=8, border_bottom_right_radius=8)
            # Inner Glow Core
            pygame.draw.rect(self.screen, (255, 200, 255) if g_height > 150 else (255, 150, 150), (gx_shake+10, 80, self.gate_width-20, int(g_center - 80 - (g_height/2))-5), 0)
            
            # Bottom Pipe
            bottom_y = int(g_center + (g_height/2))
            bottom_rect = (gx_shake, bottom_y + shake_y, self.gate_width, self.height - bottom_y)
            pygame.draw.rect(self.screen, gate_color, bottom_rect, 0, border_top_left_radius=8, border_top_right_radius=8)
            # Inner Glow Core
            pygame.draw.rect(self.screen, (255, 200, 255) if g_height > 150 else (255, 150, 150), (gx_shake+10, bottom_y + shake_y + 5, self.gate_width-20, self.height - bottom_y), 0)

        # Draw Particles (Behind player)
        for part in self.particles:
            alpha = max(0, int(255 * (part[3] / part[4])))
            px, py = int(part[0][0] + shake_x), int(part[0][1] + shake_y)
            pygame.draw.circle(self.screen, part[2], (px, py), 3)

        # Draw Player (Cyan Orb / Ship)
        px, py = int(self.player_x + shake_x), int(self.player_y + shake_y)
        
        # Tilt angle based on velocity!
        tilt = int(self.player_dy * 2) 
        
        # Draw thruster flame if pressing space
        if self.player_dy < 0:
            pygame.draw.ellipse(self.screen, (255, 150, 0), (px - 25, py - 5 + tilt, 20, 10))

        # Main Hull
        pygame.draw.circle(self.screen, (0, 150, 150), (px, py), self.player_radius + 4) # Glow Outline
        pygame.draw.circle(self.screen, (0, 255, 255), (px, py), self.player_radius)
        # Eye / Cockpit
        pygame.draw.circle(self.screen, (255, 255, 255), (px + 6, py + tilt//2), 6)

        self.draw_hud()
        pygame.display.flip()

    def quit(self):
        pygame.quit()

# Standardized entry point
AdaptiveGame = AscensionGame
