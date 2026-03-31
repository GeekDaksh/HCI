"""
run_rl.py — Train and Evaluate the Adaptive Difficulty RL System
=================================================================
Ties together all components:
  - Loads best trained EEG model (TCN or Transformer from results/)
  - Creates workload source (choose mode: replay / simulate)
  - Trains DQN agent across multiple episodes
  - Evaluates and plots the closed-loop system behaviour

Run order (after your ML models are done):
  python run_rl.py --mode replay --subject S01 --game G1 --train
  python run_rl.py --mode replay --subject S01 --game G1 --eval
  python run_rl.py --mode simulate --train

Arguments:
  --mode     replay | simulate | live
  --subject  subject ID for replay (e.g. S01)
  --game     game ID for replay (e.g. G1)
  --train    run training loop
  --eval     run evaluation with charts
  --episodes number of training episodes (default 300)
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))           # RL/ folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # HCI/ root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))  # HCI/models/ — must come first so models/transformer_model.py (77 features) is found before HCI/transformer_model.py (116 features)

from game_env        import AdaptiveGameEnv, TARGET_WORKLOAD
from workload_sources import create_workload_source
from rl_agent        import DQNAgent
from dataset_loader  import load_dataset

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

AGENT_PATH  = os.path.join(RESULTS_DIR, "rl_agent.pt")
# Force CPU for RL pipeline — the DQN is a tiny 2-layer MLP, MPS gives no
# meaningful speedup and causes tensor device mismatch errors with the
# Transformer inference inside workload_sources.py.
# MPS is only beneficial for the full model training scripts.
DEVICE      = torch.device("cpu")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_best_model(preferred=None):
    """
    Load the best performing ML model WITH trained weights.

    Priority: Transformer (best reliability, fewest extreme errors)
              → TCN (best raw accuracy) → BiLSTM+Attention

    Override with preferred='tcn' | 'transformer' | 'bilstm' from CLI.
    """
    # Model order: Transformer first for RL (lowest Low<->High confusion)
    model_candidates = [
        ("transformer", "Transformer",      "transformer_model.py"),
        ("tcn",         "TCN",              "tcn_model.py"),
        ("bilstm",      "BiLSTM+Attention", "bilstm_model.py"),
    ]

    # If user specified a model, put it first
    if preferred:
        model_candidates.sort(key=lambda x: 0 if x[0] == preferred.lower() else 1)

    for module_name, display_name, _ in model_candidates:
        try:
            if module_name == "transformer":
                from transformer_model import (EEGTransformer, D_MODEL, N_HEADS,
                                               N_LAYERS, D_FF, DROPOUT, N_CLASSES)
                model = EEGTransformer(input_dim=77, d_model=D_MODEL,
                                       n_heads=N_HEADS, n_layers=N_LAYERS,
                                       d_ff=D_FF, dropout=DROPOUT,
                                       n_classes=N_CLASSES).to(DEVICE)
                weights_path = os.path.join(RESULTS_DIR, "transformer_weights.pt")

            elif module_name == "tcn":
                from tcn_model import TCN, TCN_CHANNELS, KERNEL_SIZE, DROPOUT, N_CLASSES
                model = TCN(input_dim=77, tcn_channels=TCN_CHANNELS,
                            kernel_size=KERNEL_SIZE, dropout=DROPOUT,
                            n_classes=N_CLASSES).to(DEVICE)
                weights_path = os.path.join(RESULTS_DIR, "tcn_weights.pt")

            elif module_name == "bilstm":
                from bilstm_model import BiLSTMAttention, HIDDEN_DIM, NUM_LAYERS, DROPOUT, N_CLASSES
                model = BiLSTMAttention(input_dim=77, hidden_dim=HIDDEN_DIM,
                                        num_layers=NUM_LAYERS, dropout=DROPOUT,
                                        n_classes=N_CLASSES).to(DEVICE)
                weights_path = os.path.join(RESULTS_DIR, "bilstm_weights.pt")

            # Load trained weights if available
            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
                print(f"  Loaded {display_name} with trained weights from {weights_path}")
            else:
                print(f"  [WARN] No saved weights at {weights_path}")
                print(f"  [WARN] Model running with RANDOM weights — add save_model_weights()")
                print(f"  [WARN] to your {module_name}_model.py and re-run it first.")

            model.eval()
            return model, display_name

        except ImportError:
            continue

    raise RuntimeError(
        "No model module found. Ensure transformer_model.py, tcn_model.py, "
        "or bilstm_model.py is in the same directory."
    )


def build_scaler():
    """
    Returns a fitted StandardScaler on the full dataset.

    NOTE: The windows in windows/*.npz are already z-scored per subject
    by preprocess.py. The scaler here is kept for the live EEG mode only,
    where raw features from the headset need scaling before model inference.
    For replay mode, workload_sources.py skips scaler.transform() since
    the windows are already normalised.
    """
    X, _, _, _, _, _ = load_dataset()
    scaler = StandardScaler()
    scaler.fit(X)
    print(f"  Scaler fitted on {len(X):,} windows  (used for live mode only)")
    return scaler


# ── Training loop ─────────────────────────────────────────────────────────────

def _get_all_sessions(windows_dir="windows", dataset="all"):
    """
    Return list of all (subject, game) pairs from windows/ folder.

    dataset: "all"     — GAMEEMO + DREAMER mixed
             "gameemo" — GAMEEMO only (S01-S28)
             "dreamer" — DREAMER only (DREAMER_S01-S23)
    """
    import re
    sessions = []
    for fname in sorted(os.listdir(windows_dir)):
        if not fname.endswith(".npz"):
            continue
        # Match both GAMEEMO (S01_G1.npz) and DREAMER (DREAMER_S01_V01.npz)
        m = re.match(r"(.+)_(G\d+|V\d+)\.npz", fname)
        if not m:
            continue
        subj, game = m.group(1), m.group(2)
        is_dreamer = subj.startswith("DREAMER")
        if dataset == "dreamer" and not is_dreamer:
            continue
        if dataset == "gameemo" and is_dreamer:
            continue
        sessions.append((subj, game))
    return sessions


def train(mode, subject, game, n_episodes, model, scaler, dataset="dreamer"):
    """
    Train the DQN agent across n_episodes.

    For replay mode: cycles through ALL available sessions randomly each
    episode. Training on a single session (same 395 windows every episode)
    gives the agent no variety — it can't learn a general policy.
    Cycling through all 112 sessions exposes the agent to the full range
    of workload profiles across all 28 subjects and 4 games.
    """
    print(f"\n{'='*60}")
    print(f"  RL Training — mode={mode}  episodes={n_episodes}")
    print(f"  State: [difficulty, workload]  Actions: {{Dec, Maintain, Inc}}")
    print(f"  Reward: R = -|W_t - {TARGET_WORKLOAD}|  (flow target)")

    # For replay: build session pool for variety
    if mode == "replay":
        sessions = _get_all_sessions("windows", dataset=dataset)
        import random as _random
        print(f"  Sessions pool: {len(sessions)} sessions  (dataset={dataset})")
    print(f"{'='*60}\n")

    agent = DQNAgent(n_states=2, n_actions=3, device=DEVICE)

    all_rewards    = []
    all_flow_pct   = []
    all_final_diff = []

    for episode in range(n_episodes):

        # Rotate through all sessions for replay — variety is essential
        if mode == "replay":
            ep_subject, ep_game = _random.choice(sessions)
        else:
            ep_subject, ep_game = subject, game

        source = create_workload_source(
            mode=mode, model=model, scaler=scaler,
            subject=ep_subject, game=ep_game, windows_dir="windows"
        )
        env   = AdaptiveGameEnv(source)
        state = env.reset()

        episode_reward = 0.0
        steps_in_flow  = 0
        losses         = []

        while not env.done:
            action                   = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)

            agent.push_experience(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            episode_reward += reward
            if info["flow_zone"]:
                steps_in_flow += 1
            state = next_state

        agent.end_episode(episode_reward)

        flow_pct = steps_in_flow / max(env.step_count, 1) * 100
        all_rewards.append(episode_reward)
        all_flow_pct.append(flow_pct)
        all_final_diff.append(env.difficulty)

        if (episode + 1) % 25 == 0 or episode == 0:
            mean_r    = np.mean(all_rewards[-25:])
            mean_flow = np.mean(all_flow_pct[-25:])
            mean_loss = np.mean(losses) if losses else 0
            print(f"  Ep {episode+1:>4}/{n_episodes}  "
                  f"reward={mean_r:>7.2f}  "
                  f"flow={mean_flow:>5.1f}%  "
                  f"ε={agent.epsilon:.3f}  "
                  f"loss={mean_loss:.4f}")

    agent.save(AGENT_PATH)
    print(f"\n  Training complete. Agent saved → {AGENT_PATH}")

    _plot_training(all_rewards, all_flow_pct, mode)
    return agent


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(mode, subject, game, model, scaler):
    """
    Run one evaluation episode with the trained agent (ε=0, fully greedy).
    Produces the main closed-loop system chart.
    """
    if not os.path.exists(AGENT_PATH):
        print(f"  [WARN] No trained agent at {AGENT_PATH} — running untrained")
        agent = DQNAgent(n_states=2, n_actions=3, device=DEVICE)
    else:
        agent = DQNAgent(n_states=2, n_actions=3, device=DEVICE)
        agent.load(AGENT_PATH)

    source = create_workload_source(
        mode=mode, model=model, scaler=scaler,
        subject=subject, game=game, windows_dir="windows"
    )
    env   = AdaptiveGameEnv(source)
    state = env.reset()

    print(f"\n  Evaluation — mode={mode}  subject={subject}  game={game}")
    print(f"  {'Step':>4}  {'Workload':>9}  {'Difficulty':>10}  {'Action':>10}  {'Reward':>7}  Flow")

    steps_in_flow = 0
    while not env.done:
        action                     = agent.select_action(state, training=False)
        next_state, reward, done, info = env.step(action)

        if env.step_count % 20 == 0 or env.step_count <= 3:
            action_names = {0:"Decrease", 1:"Maintain", 2:"Increase"}
            print(f"  {env.step_count:>4}  "
                  f"{info['workload_smooth']:>9.3f}  "
                  f"{info['difficulty']:>10.3f}  "
                  f"{action_names[action]:>10}  "
                  f"{reward:>7.3f}  "
                  f"{'YES' if info['flow_zone'] else '   '}")

        if info["flow_zone"]:
            steps_in_flow += 1
        state = next_state

    flow_pct = steps_in_flow / env.step_count * 100
    mean_r   = np.mean(env.history["reward"])
    print(f"\n  Steps: {env.step_count}  "
          f"Flow zone: {steps_in_flow}/{env.step_count} ({flow_pct:.1f}%)  "
          f"Mean reward: {mean_r:.3f}")

    _plot_evaluation(env.history, mode, subject, game)


# ── Charts ────────────────────────────────────────────────────────────────────

def _plot_training(rewards, flow_pcts, mode):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    fig.suptitle(f"RL Training — Adaptive Game Difficulty  (mode={mode})",
                 fontsize=14, fontweight="bold")

    # Smooth with rolling mean
    window = min(25, len(rewards))
    r_smooth = np.convolve(rewards, np.ones(window)/window, mode="valid")
    f_smooth = np.convolve(flow_pcts, np.ones(window)/window, mode="valid")
    x = np.arange(len(r_smooth)) + window - 1

    ax = axes[0]
    ax.plot(rewards,  color="#D3D1C7", alpha=0.4, linewidth=0.7)
    ax.plot(x, r_smooth, color="#534AB7", linewidth=2)
    ax.set_ylabel("Episode reward")
    ax.set_title("Cumulative reward per episode (purple = 25-ep rolling mean)")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)

    ax = axes[1]
    ax.plot(flow_pcts, color="#9FE1CB", alpha=0.4, linewidth=0.7)
    ax.plot(x, f_smooth, color="#1D9E75", linewidth=2)
    ax.set_ylabel("% steps in flow zone")
    ax.set_xlabel("Episode")
    ax.set_title("Time in flow zone per episode (|W − W*| ≤ 0.1)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.axhline(50, color="red", linestyle="--", linewidth=0.8,
               alpha=0.5, label="50% threshold")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "rl_training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training chart → {out}")


def _plot_evaluation(history, mode, subject, game, filename="rl_evaluation.png"):
    steps = np.arange(len(history["workload_smooth"]))
    fig   = plt.figure(figsize=(16, 12))
    subtitle = f"subject={subject}  game={game}" if subject else "synthetic workload"
    fig.suptitle(f"Closed-Loop Adaptive System — {mode} mode  ({subtitle})",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Workload over time ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(steps,
                     TARGET_WORKLOAD - 0.1, TARGET_WORKLOAD + 0.1,
                     alpha=0.2, color="#1D9E75", label="Flow zone")
    ax1.plot(steps, history["workload_raw"],    color="#D3D1C7",
             alpha=0.5, linewidth=0.8, label="Workload (raw)")
    ax1.plot(steps, history["workload_smooth"], color="#534AB7",
             linewidth=2, label="Workload (smoothed)")
    ax1.axhline(TARGET_WORKLOAD, color="#1D9E75", linestyle="--",
                linewidth=1.5, label=f"Target W*={TARGET_WORKLOAD}")
    ax1.set_ylabel("Workload W_t  [0=calm, 1=overloaded]")
    ax1.set_xlabel("Step (1 step = 2 seconds)")
    ax1.set_title("Workload trajectory vs flow target")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1)

    # ── 2. Difficulty over time ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    difficulty = history["difficulty"]
    ax2.plot(steps, difficulty, color="#D85A30", linewidth=2)
    ax2.fill_between(steps, difficulty, 0.5, alpha=0.15,
                     color="#D85A30", label="Difficulty vs medium")
    for thresh, label, color in [(0.3,"Easy","#639922"), (0.6,"Hard","#D85A30"),
                                  (0.8,"Extreme","#A32D2D")]:
        ax2.axhline(thresh, linestyle=":", alpha=0.5, color=color,
                    linewidth=1, label=label)
    ax2.set_ylabel("Difficulty D_t  [0=easiest, 1=hardest]")
    ax2.set_xlabel("Step")
    ax2.set_title("Game difficulty adjustment by RL agent")
    ax2.legend(fontsize=8, ncol=4); ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    # ── 3. Actions histogram ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    action_labels = ["Decrease", "Maintain", "Increase"]
    action_counts = [history["action"].count(i) for i in range(3)]
    bars = ax3.bar(action_labels, action_counts,
                   color=["#378ADD","#888780","#D85A30"],
                   alpha=0.85, edgecolor="white")
    ax3.set_ylabel("Count")
    ax3.set_title("Action distribution")
    ax3.grid(axis="y", alpha=0.3)
    for bar, cnt in zip(bars, action_counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(cnt), ha="center", fontsize=10, fontweight="bold")

    # ── 4. Reward over time ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(steps, history["reward"], color="#1D9E75", linewidth=1, alpha=0.7)
    window = min(20, len(history["reward"]))
    r_smooth = np.convolve(history["reward"],
                           np.ones(window)/window, mode="same")
    ax4.plot(steps, r_smooth, color="#085041", linewidth=2,
             label=f"{window}-step rolling mean")
    ax4.axhline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax4.set_ylabel("Reward R = −|W_t − W*|")
    ax4.set_xlabel("Step")
    ax4.set_title("Per-step reward")
    ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

    # Summary stats as text
    flow_pct = sum(1 for r in history["reward"] if r > -0.1) / len(history["reward"]) * 100
    fig.text(0.5, 0.01,
             f"Steps: {len(steps)}   "
             f"Flow zone: {flow_pct:.1f}%   "
             f"Mean reward: {np.mean(history['reward']):.3f}   "
             f"Final difficulty: {difficulty[-1]:.2f}",
             ha="center", fontsize=11, color="#444441")

    out = os.path.join(RESULTS_DIR, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Evaluation chart → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train/evaluate RL adaptive difficulty agent"
    )
    parser.add_argument("--mode",     default="simulate",
                        choices=["live", "replay", "simulate"],
                        help="Workload source mode")
    parser.add_argument("--subject",  default=None,
                        help="Subject ID for replay (e.g. S01)")
    parser.add_argument("--game",     default=None,
                        help="Game ID for replay (e.g. G1)")
    parser.add_argument("--train",    action="store_true",
                        help="Run training loop")
    parser.add_argument("--eval",     action="store_true",
                        help="Run evaluation")
    parser.add_argument("--model",    default=None,
                        choices=["transformer", "tcn", "bilstm"],
                        help="Which EEG model to use (default: transformer)")
    parser.add_argument("--episodes", type=int, default=300,
                        help="Number of training episodes")
    parser.add_argument("--dataset", default="dreamer",
                        choices=["all", "gameemo", "dreamer"],
                        help="Session pool for replay training (default: dreamer)")
    args = parser.parse_args()

    if not args.train and not args.eval:
        print("Specify --train and/or --eval")
        parser.print_help()
        return

    print(f"Device: {DEVICE}")
    print(f"Mode  : {args.mode}")

    # Load EEG model (only needed for replay/live modes)
    model, model_name = None, None
    scaler = None
    if args.mode in ("replay", "live"):
        print("\nLoading EEG model...")
        model, model_name = load_best_model(preferred=args.model)
        print("Fitting scaler on dataset...")
        scaler = build_scaler()

    if args.train:
        train(args.mode, args.subject, args.game,
              args.episodes, model, scaler,
              dataset=getattr(args, "dataset", "dreamer"))

    if args.eval:
        evaluate(args.mode, args.subject, args.game, model, scaler)


if __name__ == "__main__":
    main()
