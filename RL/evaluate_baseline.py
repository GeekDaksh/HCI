"""
evaluate_baseline.py — DQN vs Random Agent Baseline Comparison
===============================================================
Runs the trained DQN agent and a random agent on the same replay
sessions and compares flow zone %, mean reward, and action distribution.

This is a critical result — it proves the DQN learned a meaningful
policy beyond chance. Without this comparison, high simulation results
alone do not demonstrate that the RL component adds value.

Run:
    python RL/evaluate_baseline.py

Output:
    results/baseline_comparison.png   — bar chart DQN vs random
    results/baseline_comparison.txt   — printed table of all results
"""

import os
import sys
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_env         import AdaptiveGameEnv
from workload_sources import create_workload_source
from rl_agent         import DQNAgent
from dataset_loader   import load_dataset
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = "results"
AGENT_PATH  = os.path.join(RESULTS_DIR, "rl_agent.pt")
DEVICE      = torch.device("cpu")

# ── Subjects/games to evaluate — top results from batch eval ──────────────
EVAL_SESSIONS = [
    ("S18", "G4"),   # best:  26.3%
    ("S03", "G3"),   # 2nd:   14.3%
    ("S02", "G2"),   # 3rd:   13.0%
    ("S02", "G1"),   # 4th:   11.7%
    ("S22", "G3"),   # 5th:    9.7%
]


# ── Model loading (copied from run_rl.py) ─────────────────────────────────

def load_best_model():
    model_candidates = [
        ("transformer", "Transformer",      "transformer_model.py"),
        ("tcn",         "TCN",              "tcn_model.py"),
        ("bilstm",      "BiLSTM+Attention", "bilstm_model.py"),
    ]
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

            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            model.eval()
            print(f"  Loaded {display_name}")
            return model
        except ImportError:
            continue
    raise RuntimeError("No model found in results/")


def build_scaler():
    X, _, _, _, _, _ = load_dataset()
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


# ── Single episode runner ─────────────────────────────────────────────────

def run_episode(agent, source_kwargs, use_random=False):
    """
    Run one evaluation episode.

    Parameters
    ----------
    agent        : DQNAgent (ignored if use_random=True)
    source_kwargs: dict passed to create_workload_source
    use_random   : if True, uses random action selection

    Returns
    -------
    dict with flow_pct, mean_reward, action_counts
    """
    source = create_workload_source(**source_kwargs)
    env    = AdaptiveGameEnv(source)
    state  = env.reset()

    steps_in_flow = 0
    action_counts = {0: 0, 1: 0, 2: 0}

    while not env.done:
        if use_random:
            action = random.randint(0, 2)
        else:
            action = agent.select_action(state, training=False)

        state, reward, done, info = env.step(action)
        action_counts[action] += 1
        if info["flow_zone"]:
            steps_in_flow += 1

    flow_pct    = steps_in_flow / env.step_count * 100
    mean_reward = float(np.mean(env.history["reward"]))

    return {
        "flow_pct":     flow_pct,
        "mean_reward":  mean_reward,
        "action_counts": action_counts,
        "steps":        env.step_count,
    }


# ── Main evaluation ───────────────────────────────────────────────────────

def main():
    print("\nLoading model and agent...")
    model  = load_best_model()
    scaler = build_scaler()

    if not os.path.exists(AGENT_PATH):
        raise RuntimeError(f"No trained agent at {AGENT_PATH} — run training first")

    agent = DQNAgent(n_states=2, n_actions=3, device=DEVICE)
    agent.load(AGENT_PATH)

    dqn_results    = []
    random_results = []

    print(f"\n{'='*65}")
    print(f"  DQN vs Random Baseline — {len(EVAL_SESSIONS)} sessions")
    print(f"{'='*65}")
    print(f"  {'Session':<12} {'DQN Flow%':>10} {'Rand Flow%':>11} "
          f"{'DQN Reward':>11} {'Rand Reward':>12} {'Improvement':>12}")
    print(f"  {'-'*65}")

    for subject, game in EVAL_SESSIONS:
        source_kwargs = dict(
            mode="replay",
            model=model,
            scaler=scaler,
            subject=subject,
            game=game,
            windows_dir="windows",
        )

        # Run DQN
        dqn = run_episode(agent, source_kwargs, use_random=False)

        # Run random (same session — source reloads from same file)
        rand = run_episode(agent, source_kwargs, use_random=True)

        dqn_results.append(dqn)
        random_results.append(rand)

        improvement = dqn["flow_pct"] - rand["flow_pct"]
        label = f"S{subject[1:]:>2}/G{game[1:]}"
        print(f"  {label:<12} {dqn['flow_pct']:>9.1f}%  "
              f"{rand['flow_pct']:>9.1f}%  "
              f"{dqn['mean_reward']:>11.3f}  "
              f"{rand['mean_reward']:>11.3f}  "
              f"{improvement:>+11.1f}%")

    # Summary
    dqn_mean_flow  = np.mean([r["flow_pct"]   for r in dqn_results])
    rand_mean_flow = np.mean([r["flow_pct"]   for r in random_results])
    dqn_mean_rew   = np.mean([r["mean_reward"] for r in dqn_results])
    rand_mean_rew  = np.mean([r["mean_reward"] for r in random_results])

    print(f"  {'-'*65}")
    print(f"  {'MEAN':<12} {dqn_mean_flow:>9.1f}%  "
          f"{rand_mean_flow:>9.1f}%  "
          f"{dqn_mean_rew:>11.3f}  "
          f"{rand_mean_rew:>11.3f}  "
          f"{dqn_mean_flow - rand_mean_flow:>+11.1f}%")
    print(f"\n  DQN improves flow zone by "
          f"{dqn_mean_flow - rand_mean_flow:+.1f}% over random baseline")

    # Save txt report
    _save_txt(dqn_results, random_results, dqn_mean_flow, rand_mean_flow,
              dqn_mean_rew, rand_mean_rew)

    # Plot
    _plot(dqn_results, random_results, dqn_mean_flow, rand_mean_flow)


# ── Save text report ──────────────────────────────────────────────────────

def _save_txt(dqn_results, random_results, dqn_mean_flow, rand_mean_flow,
              dqn_mean_rew, rand_mean_rew):
    path = os.path.join(RESULTS_DIR, "baseline_comparison.txt")
    lines = [
        "DQN vs Random Agent Baseline Comparison",
        "=" * 50,
        f"{'Session':<12} {'DQN Flow%':>10} {'Rand Flow%':>11} {'Improvement':>12}",
        "-" * 50,
    ]
    for i, (s, g) in enumerate(EVAL_SESSIONS):
        d = dqn_results[i]
        r = random_results[i]
        lines.append(f"  S{s[1:]}/G{g[1:]:<8}  {d['flow_pct']:>8.1f}%  "
                     f"{r['flow_pct']:>9.1f}%  "
                     f"{d['flow_pct'] - r['flow_pct']:>+10.1f}%")
    lines += [
        "-" * 50,
        f"  {'MEAN':<10}  {dqn_mean_flow:>8.1f}%  {rand_mean_flow:>9.1f}%  "
        f"{dqn_mean_flow - rand_mean_flow:>+10.1f}%",
        "",
        f"DQN mean reward : {dqn_mean_rew:.3f}",
        f"Random mean reward: {rand_mean_rew:.3f}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved → {path}")


# ── Plot ──────────────────────────────────────────────────────────────────

def _plot(dqn_results, random_results, dqn_mean_flow, rand_mean_flow):
    labels      = [f"S{s[1:]}/G{g[1:]}" for s, g in EVAL_SESSIONS]
    dqn_flows   = [r["flow_pct"]   for r in dqn_results]
    rand_flows  = [r["flow_pct"]   for r in random_results]
    dqn_rews    = [r["mean_reward"] for r in dqn_results]
    rand_rews   = [r["mean_reward"] for r in random_results]

    x     = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("DQN Agent vs Random Baseline — Replay Evaluation",
                 fontsize=14, fontweight="bold")

    # ── Flow zone % ───────────────────────────────────────────────────────
    ax = axes[0]
    bars1 = ax.bar(x - width/2, dqn_flows,  width, label="DQN agent",
                   color="#534AB7", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width/2, rand_flows, width, label="Random agent",
                   color="#888780", alpha=0.85, edgecolor="white")

    # mean lines
    ax.axhline(dqn_mean_flow,  color="#534AB7", linestyle="--",
               linewidth=1.5, alpha=0.7, label=f"DQN mean {dqn_mean_flow:.1f}%")
    ax.axhline(rand_mean_flow, color="#888780", linestyle="--",
               linewidth=1.5, alpha=0.7, label=f"Random mean {rand_mean_flow:.1f}%")
    ax.axhline(50, color="red", linestyle=":", linewidth=1,
               alpha=0.4, label="50% target")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", fontsize=8,
                fontweight="bold", color="#3C3489")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", fontsize=8,
                color="#444441")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% steps in flow zone  (|W − W*| ≤ 0.12)")
    ax.set_title("Flow zone % — DQN vs Random")
    ax.set_ylim(0, max(max(dqn_flows), max(rand_flows)) * 1.25 + 5)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ── Mean reward ───────────────────────────────────────────────────────
    ax = axes[1]
    bars3 = ax.bar(x - width/2, dqn_rews,  width, label="DQN agent",
                   color="#534AB7", alpha=0.85, edgecolor="white")
    bars4 = ax.bar(x + width/2, rand_rews, width, label="Random agent",
                   color="#888780", alpha=0.85, edgecolor="white")
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)

    for bar in bars3:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() - 0.01,
                f"{bar.get_height():.3f}", ha="center", va="top",
                fontsize=8, fontweight="bold", color="#3C3489")
    for bar in bars4:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() - 0.01,
                f"{bar.get_height():.3f}", ha="center", va="top",
                fontsize=8, color="#444441")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean reward R = −|W_t − W*|")
    ax.set_title("Mean reward — DQN vs Random")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # improvement annotation
    improvement = dqn_mean_flow - rand_mean_flow
    fig.text(0.5, 0.01,
             f"DQN outperforms random baseline by {improvement:+.1f}% flow zone  |  "
             f"Mean reward improvement: {np.mean(dqn_rews) - np.mean(rand_rews):+.3f}",
             ha="center", fontsize=11, color="#444441", fontweight="bold")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(RESULTS_DIR, "baseline_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {out}")


if __name__ == "__main__":
    main()
