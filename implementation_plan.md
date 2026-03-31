# Research-Backed PyGame Simulation Engine for Closed-Loop BCI

Given the hardware constraints (no EEG headsets or IRL testing time), the most legitimate and industry-respected alternative is building a **"Digital Twin" Co-Simulation Framework**. 

In professional BCI and neuroergonomics research, when real-time clinical testing is not viable, researchers build a highly polished offline simulation. This proves the *entire software architecture* works seamlessly. You will simulate the closed loop via a **PyGame visual engine** that reacts to the RL Agent, driven by your existing EEG datasets and the Transformer pipeline.

## User Review Required

> [!WARNING]
> Please review this plan carefully. This represents the final capstone "deliverable" of your project, shifting from hardware testing to a fully software-based Digital Twin evaluation. If you approve, I will begin writing the PyGame engine and the integration loop.

## The Architecture: Replay-Driven Digital Twin

We will build a visual PyGame simulation that links the three massive components you've already built:
1. **The Data Pipeline** (GAMEEMO preprocessed windows)
2. **The Predictive Pipeline** (Transformer extracting continuous `W` from 77 engineered features)
3. **The Control Pipeline** (DQN Agent mapping `W` into Game Difficulty `D`)

### How it will work in real-time:
Instead of running a dry console `run_rl.py` script, we will create `simulate_pygame_loop.py`. 
- **The Visuals**: A python game (e.g., an endless dodger/shooter) running at 60 FPS.
- **The Physics**: The game's `enemy speed`, `spawn rate`, and `complexity` are strictly controlled by a `Difficulty` variable $D \in [0,1]$.
- **The Brain**: On a separate thread or timer (every 2 seconds, mimicking the 2s EEG window), the engine polls the `ReplayWorkloadSource` or `SimulationWorkloadSource`. 
- **The RL Agent**: The DQN takes the predicted Workload and current Difficulty, outputs an Action (Increase, Maintain, Decrease), and passes the new Difficulty to the Game Engine.

## Proposed Changes

---

### Phase 1: The Adaptive PyGame Engine
We will create a robust PyGame application that strictly consumes the `game_env.py` parameters.

#### [NEW] `RL/adaptive_game.py`
- Implements a 2D interactive game (e.g., dodging asteroids or enemies).
- Contains a `update_difficulty(difficulty_dict)` method reflecting `enemy_speed`, `spawn_rate`, and `obstacle_density`.
- Includes an **On-Screen Dashboard** (HUD) updating live:
     - Real-time Workload line chart (rolling window or bar)
     - Target Flow Zone boundaries (0.6)
     - Current DQN Action (`Decrease`, `Maintain`, `Increase`)
     - Real-time Difficulty gauge

### Phase 2: The Simulation Bridge
We need a runner script that bridges PyGame, the RL Agent, and the Replay/Simulator modes.

#### [NEW] `RL/run_simulation.py`
- Sets up the `AdaptiveGameEnv` from your `game_env.py`.
- Loads the best pre-trained DQN model and the Transformer best weights.
- Runs the PyGame event loop.
- Triggers an RL step every $X$ frames (e.g., every 60 frames = 1 "window" step).
- Passes the environment `info` dictionary (Workload, Flow status) directly to the PyGame HUD.

### Phase 3: Simulated "Digital Twin" Validation
Since the EEG `ReplayMode` is prerecorded, the Replay human won't *react* to the difficulty changes. To demonstrate a *perfectly closed, responsive loop*, we will enhance the `SimulationWorkloadSource`.
#### [MODIFY] `RL/workload_sources.py`
- Tweak `SimulationWorkloadSource` to ensure the mathematical response curve (where Workload increases as Game Difficulty increases) is realistic and heavily resembles the GAMEEMO variance. This creates a perfect "Digital Twin" of a human player for demonstration purposes.

## Benefits for The Viva & Industry

1. **High Visual Impact**: A live PyGame window reacting dynamically to Neural Data (even Replayed data) is visually stunning.
2. **Architectural Validity**: You prove that your pipeline is modular. You successfully decoupled the signal acquisition (EEG) from the actuation (Game). 
3. **Safe Evaluation**: You avoid the unreliability of bluetooth dropouts or noisy raw sensors during presentations. The simulation proves mathematical correctness of your RL algorithm and your EEG models handling pre-recorded data beautifully.

## Open Questions

> [!IMPORTANT]
> 1. Do you have a preference for the *type* of game? An endless scroller (asteroids/racing), an aim-trainer, or a simple survival dodger?
> 2. Would you like me to start by creating the base PyGame script and the `run_simulation.py` integration loop?

## Verification Plan

### Automated/Mathematical Verification
- Ensure PyGame FPS does not drop during the PyTorch Transformer inference step.
- Ensure the difficulty mathematically smoothly shifts without erratic game state snapping.

### Manual Verification
- Run the GUI simulation via `python RL/run_simulation.py --mode simulation` and visually confirm that the game becomes visually harder/faster as difficulty goes up, and that the RL agent successfully clamps the workload.
