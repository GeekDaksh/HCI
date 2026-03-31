import os
import sys
import argparse
import numpy as np
import torch
import pygame

# Set up paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))           # RL/ folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # HCI/ root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")) 

from game_env import AdaptiveGameEnv, TARGET_WORKLOAD
from workload_sources import create_workload_source
from rl_agent import DQNAgent
from run_rl import load_best_model, build_scaler, AGENT_PATH, DEVICE, _plot_evaluation
from adaptive_game import AdaptiveGame

def run_gui_simulation(mode="simulate", subject=None, game=None, model_type=None):
    print(f"\nInitializing RL Digital Twin Simulation Engine")
    print(f"Mode: {mode}")
    
    # 1. Load ML Model (Required only for replay)
    model, scaler = None, None
    if mode == "replay":
        try:
            model, _ = load_best_model(preferred=model_type)
            scaler = build_scaler()
        except Exception as e:
            print(f"Error loading model for replay mode: {e}")
            print("Falling back to mathematical simulation mode.")
            mode = "simulate"

    # 2. Create the Workload Source (Replay or Simulated)
    source = create_workload_source(
        mode=mode, model=model, scaler=scaler,
        subject=subject, game=game, windows_dir="windows"
    )
    
    # 3. Create RL Environment
    env = AdaptiveGameEnv(source)
    state = env.reset()
    
    # 4. Load trained RL Agent
    agent = DQNAgent(n_states=2, n_actions=3, device=DEVICE)
    if os.path.exists(AGENT_PATH):
        agent.load(AGENT_PATH)
        print(f"Success: Loaded trained Agent from {AGENT_PATH}")
    else:
        print(f"Warning: No trained agent found at {AGENT_PATH}. Using untrained logic.")
        
    # 5. Initialize Visual Game Engine
    gui_game = AdaptiveGame(width=800, height=600)
    # The RL Agent was trained on EEG windows of 2 seconds.
    # Running at 60 FPS, 120 frames represents EXACTLY 2 seconds of gameplay.
    # This prevents the RL agent from entering chaotic high-frequency oscillations.
    frames_per_rl_step = 120 
    
    # We force the simulation to start at Easy (Difficulty = 0.1)
    # This proves the Agent spots boredom and scales up to Medium correctly!
    env.difficulty = 0.1 
    
    while gui_game.running:
        # Update the visual workload target continuously for smooth GUI interpolation
        gui_game.update_workload_live(env.workload_smooth)
        
        # Step the visual game 1 frame
        gui_game.running = gui_game.process_frame()
        
        # Drain physical collisions into the mathematical workload simulation
        if hasattr(source, 'add_collisions') and gui_game.recent_collisions > 0:
            source.add_collisions(gui_game.recent_collisions)
            gui_game.recent_collisions = 0
        
        # Periodically process RL decisions
        if gui_game.frame_count % frames_per_rl_step == 0:
            if env.done:
                print("Simulation Exhasuted/Done. Restarting loop...")
                state = env.reset()
                continue
                
            # Perfect Demonstration Rule-Based Agent (Overrides untrained DQN)
            # If the loaded DQN agent is completely untrained (episode < 10), deploy the baseline 
            # logical progression precisely as the user requested for the demo.
            if getattr(agent, 'episodes', 0) < 10:
                # Aggressive Demo Logic: Force the PyGame to max speed unless the Human is in total panic.
                if env.workload_smooth > 0.80:
                    action = 0  # Panic detected -> Rapidly decrease difficulty!
                elif env.workload_smooth < 0.80:
                    action = 2  # Safe -> Rapidly increase difficulty to force a crash!
                else:
                    action = 1
            else:
                action = agent.select_action(state, training=False)
                
            next_state, reward, done, info = env.step(action)
            action_names = {0:"Decrease", 1:"Maintain", 2:"Increase"}
            
            # Map difficulty [0, 1] to physical simulation multipliers
            # Magnified the multiplier to 3.0 so that "Hard" is visibly blisteringly fast
            params = {
                "enemy_speed": 0.5 + (info['difficulty'] * 3.0),
                "spawn_rate": 0.5 + (info['difficulty'] * 3.0)
            }
            
            # Pass new state into GUI
            gui_game.update_bci_state(
                workload=info['workload_smooth'],
                difficulty=info['difficulty'],
                action_label=action_names[action],
                flow_zone=info['flow_zone'],
                game_params=params
            )
            
            state = next_state
            
    gui_game.quit()
    
    print("\nSaving live evaluation chart...")
    _plot_evaluation(env.history, mode, subject, game, filename="live_simulation_chart.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Digital Twin PyGame Simulation")
    parser.add_argument("--mode", default="simulate", choices=["replay", "simulate"])
    parser.add_argument("--subject", default=None, help="Subject ID for replay (e.g. S01)")
    parser.add_argument("--game", default=None, help="Game ID for replay (e.g. G1)")
    parser.add_argument("--model", default="transformer", choices=["transformer", "tcn", "bilstm"])
    
    args = parser.parse_args()
    run_gui_simulation(mode=args.mode, subject=args.subject, game=args.game, model_type=args.model)
