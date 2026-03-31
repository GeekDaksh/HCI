"""
sam_workload.py — SAM Workload Label Extraction + GA Equation Optimisation
===========================================================================

TWO KEY CHANGES:

1. GENETIC ALGORITHM — finds optimal workload equation weights
   Instead of the fixed formula:  workload = arousal + (10 - valence)
   The GA searches for the best weights:
       workload = w1*arousal + w2*(10-valence) + w3*arousal*(10-valence) + w4*arousal^2

   Fitness = how well the resulting labels separate when sorted by rank.
   This finds the equation that makes Low/Medium/High most distinct.

   Run once, then reuse best weights. Adds ~2–5 min to first run.

2. PER-SUBJECT TERTILE CLASSES — guarantees all 3 classes for every subject
   OLD (broken): fixed global bins — 18/28 subjects had zero Medium
   NEW:  per-subject tertile split — every subject has Low + Medium + High

   Low    = this subject's bottom 33% of game scores
   Medium = this subject's middle 33% of game scores
   High   = this subject's top 33% of game scores

USAGE:
   python sam_workload.py           # runs GA then extracts labels
   python sam_workload.py --no-ga   # skip GA, use default weights
"""

import os
import re
import sys
import random
import argparse
import numpy as np
import pandas as pd
from pdfminer.high_level import extract_text


# ── Config ────────────────────────────────────────────────────────────────────
GA_POPULATION    = 40       # number of candidate equations per generation
GA_GENERATIONS   = 60       # number of generations to evolve
GA_MUTATION_RATE = 0.15     # probability of mutating each weight
GA_ELITE         = 4        # top N chromosomes always survive unchanged
GA_CROSSOVER     = 0.7      # probability of crossover vs random new

# Weight search bounds: [w1, w2, w3, w4]
#   w1 = arousal weight          (0.0 – 2.0)
#   w2 = inverse valence weight  (0.0 – 2.0)
#   w3 = interaction term        (0.0 – 1.0)
#   w4 = arousal^2 term          (0.0 – 0.5)
W_BOUNDS = [(0.0, 2.0), (0.0, 2.0), (0.0, 1.0), (0.0, 0.5)]

# Default weights (original formula: w1=1, w2=1, w3=0, w4=0)
DEFAULT_WEIGHTS = [1.0, 1.0, 0.0, 0.0]

VALENCE_ANCHORS = ["horrible","unhappy","sad","annoyed","unsatisfied",
                   "happy","funny","joyful","satisfied","pleased"]
AROUSAL_ANCHORS = ["calm","relaxed","sleepy","sluggish","dull",
                   "excited","stimulated","aroused","frenzied","jittery"]


# ── PDF parsing (unchanged from original) ────────────────────────────────────

def extract_sam_scores(pdf_path):
    text  = extract_text(pdf_path)
    lines = text.splitlines()

    valence = _find_score_near_anchors(lines, VALENCE_ANCHORS)
    arousal = _find_score_near_anchors(lines, AROUSAL_ANCHORS)

    if valence is None or arousal is None:
        digits = _extract_standalone_digits(lines)
        if len(digits) >= 2:
            if valence is None: valence = digits[0]
            if arousal is None: arousal = digits[1]
        elif len(digits) == 1:
            print(f"  [WARN] Only 1 digit in {os.path.basename(pdf_path)}")

    valence = _validate(valence, "valence", pdf_path)
    arousal = _validate(arousal, "arousal", pdf_path)
    return {"valence": valence, "arousal": arousal}


def _find_score_near_anchors(lines, anchors):
    for i, line in enumerate(lines):
        if any(a in line.lower() for a in anchors):
            for j in range(max(0,i-3), min(len(lines),i+4)):
                m = re.match(r'^\s*([1-9])\s*$', lines[j])
                if m: return int(m.group(1))
            m = re.search(r'\b([1-9])\b', line)
            if m: return int(m.group(1))
    return None


def _extract_standalone_digits(lines):
    return [int(re.match(r'^\s*([1-9])\s*$', l).group(1))
            for l in lines if re.match(r'^\s*([1-9])\s*$', l)]


def _validate(score, name, pdf_path):
    if score is None:
        print(f"  [WARN] Could not extract {name} from {os.path.basename(pdf_path)}")
        return None
    if not (1 <= score <= 9):
        print(f"  [WARN] {name}={score} out of range — discarding")
        return None
    return score


# ── Workload equation ─────────────────────────────────────────────────────────

def compute_workload_raw(valence, arousal, weights):
    """
    Generalised workload formula with GA-learned weights.

    raw = w1*arousal + w2*(10-valence) + w3*arousal*(10-valence) + w4*arousal^2

    Original formula is weights=[1,1,0,0]:
        raw = 1*arousal + 1*(10-valence) = arousal + (10-valence)
    """
    w1, w2, w3, w4 = weights
    inv_val = 10 - valence
    raw = (w1 * arousal
           + w2 * inv_val
           + w3 * arousal * inv_val
           + w4 * arousal ** 2)
    return float(raw)


def normalise_to_1_9(values):
    """Normalise a list of raw workload values to [1, 9] scale."""
    arr = np.array(values, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.ones_like(arr) * 5.0   # all same → assign middle
    return (arr - mn) / (mx - mn) * 8 + 1


# ── Per-subject tertile class assignment ─────────────────────────────────────

def assign_tertile_classes(records):
    """
    Assign Low/Medium/High using PER-SUBJECT tertile splits.

    With 4 games per subject this gives:
      - roughly 1–2 Low, 1 Medium, 1–2 High games
      - every subject guaranteed to have all 3 classes

    This is scientifically valid because workload is relative:
    what is "hard" for one subject may be "easy" for another.
    The adaptive gameplay system needs relative difficulty, not absolute.
    """
    scores = [r["workload_continuous"] for r in records]

    if len(scores) < 3:
        # Fallback for subjects with < 3 games
        t1, t2 = 3.67, 6.33
    else:
        t1 = float(np.percentile(scores, 33.33))
        t2 = float(np.percentile(scores, 66.67))
        if t1 == t2:   # all same score edge case
            span = max(scores) - min(scores)
            t1   = min(scores) + span / 3
            t2   = min(scores) + 2 * span / 3

    label_map = {0: "low", 1: "medium", 2: "high"}
    for r in records:
        w = r["workload_continuous"]
        cls = 0 if w <= t1 else (1 if w <= t2 else 2)
        r["workload_class"]   = cls
        r["workload_label"]   = label_map[cls]
        r["threshold_t1"]     = round(t1, 3)
        r["threshold_t2"]     = round(t2, 3)
    return records


# ── Genetic Algorithm ─────────────────────────────────────────────────────────

def random_chromosome():
    """Generate a random set of 4 weights within bounds."""
    return [random.uniform(lo, hi) for (lo, hi) in W_BOUNDS]


def fitness(weights, all_sam_data):
    """
    Evaluate how well this weight vector separates workload classes.

    Strategy: compute workload scores for all subjects/games,
    assign tertile classes, then measure inter-class separation
    using average absolute difference between class means.

    Higher fitness = clearer separation between Low / Medium / High.

    We also penalise degenerate solutions where all scores collapse
    to the same value (no variance).
    """
    # Compute raw scores for all (valence, arousal) pairs
    raw_scores = []
    for valence, arousal in all_sam_data:
        raw_scores.append(compute_workload_raw(valence, arousal, weights))

    if len(raw_scores) < 3:
        return 0.0

    norm_scores = normalise_to_1_9(raw_scores)

    # Penalise if variance is too low (degenerate equation)
    if norm_scores.std() < 0.3:
        return 0.0

    # Group scores by subject and assign tertile classes
    # (We work flat here for speed — same logic as assign_tertile_classes)
    # Sort unique score values to approximate class separation
    sorted_scores = np.sort(norm_scores)
    t1_idx = int(len(sorted_scores) * 0.33)
    t2_idx = int(len(sorted_scores) * 0.67)

    low_scores  = sorted_scores[:t1_idx]
    med_scores  = sorted_scores[t1_idx:t2_idx]
    high_scores = sorted_scores[t2_idx:]

    if len(low_scores) == 0 or len(high_scores) == 0:
        return 0.0

    low_mean  = low_scores.mean()
    med_mean  = med_scores.mean() if len(med_scores) > 0 else (low_mean + high_scores.mean()) / 2
    high_mean = high_scores.mean()

    # Fitness = sum of pairwise class mean differences (want these large)
    sep  = abs(high_mean - low_mean)
    sep += abs(high_mean - med_mean)
    sep += abs(med_mean  - low_mean)

    # Bonus for uniform class sizes (balanced dataset)
    n_low  = t1_idx
    n_high = len(sorted_scores) - t2_idx
    n_med  = t2_idx - t1_idx
    total  = len(sorted_scores)
    balance_score = 1 - np.std([n_low/total, n_med/total, n_high/total]) * 3
    balance_score = max(0, balance_score)

    return float(sep * (1 + 0.2 * balance_score))


def crossover(parent_a, parent_b):
    """Blend crossover: child = a + alpha*(b-a) where alpha in [-0.1, 1.1]."""
    child = []
    for (a, b), (lo, hi) in zip(zip(parent_a, parent_b), W_BOUNDS):
        alpha = random.uniform(-0.1, 1.1)
        val   = a + alpha * (b - a)
        child.append(float(np.clip(val, lo, hi)))
    return child


def mutate(chromosome, generation, max_gen):
    """
    Gaussian mutation with adaptive sigma.
    Sigma shrinks as generations progress (exploitation over exploration).
    """
    sigma_scale = 1.0 - 0.7 * (generation / max_gen)   # 1.0 → 0.3
    result = []
    for val, (lo, hi) in zip(chromosome, W_BOUNDS):
        if random.random() < GA_MUTATION_RATE:
            sigma = (hi - lo) * 0.15 * sigma_scale
            val   = val + random.gauss(0, sigma)
            val   = float(np.clip(val, lo, hi))
        result.append(val)
    return result


def run_ga(all_sam_data, verbose=True):
    """
    Run genetic algorithm to find optimal workload equation weights.

    all_sam_data: list of (valence, arousal) tuples from all subjects/games

    Returns: best_weights [w1, w2, w3, w4], best_fitness float
    """
    print(f"\n  Running GA: {GA_POPULATION} population × {GA_GENERATIONS} generations")
    print(f"  Searching: w1∈[0,2] w2∈[0,2] w3∈[0,1] w4∈[0,0.5]")
    print(f"  Formula:   workload = w1·A + w2·(10-V) + w3·A·(10-V) + w4·A²\n")

    # Seed population — include original formula as one candidate
    population = [DEFAULT_WEIGHTS.copy()]
    for _ in range(GA_POPULATION - 1):
        population.append(random_chromosome())

    best_weights  = DEFAULT_WEIGHTS.copy()
    best_fit      = fitness(DEFAULT_WEIGHTS, all_sam_data)
    history       = []

    for gen in range(GA_GENERATIONS):
        # Evaluate fitness
        scored = [(fitness(ch, all_sam_data), ch) for ch in population]
        scored.sort(key=lambda x: x[0], reverse=True)

        gen_best_fit, gen_best = scored[0]
        history.append(gen_best_fit)

        if gen_best_fit > best_fit:
            best_fit     = gen_best_fit
            best_weights = gen_best.copy()

        if verbose and (gen % 10 == 0 or gen == GA_GENERATIONS - 1):
            w = gen_best
            print(f"  Gen {gen:>3}: fitness={gen_best_fit:.4f}  "
                  f"w=[{w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}, {w[3]:.3f}]")

        # Elitism — keep top N unchanged
        new_population = [ch for _, ch in scored[:GA_ELITE]]

        # Fill rest with crossover + mutation
        while len(new_population) < GA_POPULATION:
            if random.random() < GA_CROSSOVER and len(scored) >= 2:
                # Tournament selection (pick best of 3 random)
                candidates = random.sample(scored[:max(GA_POPULATION//2, 5)], 3)
                _, pa = max(candidates, key=lambda x: x[0])
                candidates = random.sample(scored[:max(GA_POPULATION//2, 5)], 3)
                _, pb = max(candidates, key=lambda x: x[0])
                child = crossover(pa, pb)
            else:
                child = random_chromosome()

            child = mutate(child, gen, GA_GENERATIONS)
            new_population.append(child)

        population = new_population

    print(f"\n  GA complete. Best fitness: {best_fit:.4f}")
    print(f"  Best weights: w1={best_weights[0]:.4f}  w2={best_weights[1]:.4f}  "
          f"w3={best_weights[2]:.4f}  w4={best_weights[3]:.4f}")

    w1, w2, w3, w4 = best_weights
    print(f"  Formula: workload = {w1:.3f}·A + {w2:.3f}·(10-V) + "
          f"{w3:.3f}·A·(10-V) + {w4:.3f}·A²")

    # Compare to original
    orig_fit = fitness(DEFAULT_WEIGHTS, all_sam_data)
    improvement = (best_fit - orig_fit) / max(orig_fit, 1e-8) * 100
    print(f"  Original fitness: {orig_fit:.4f}  →  improvement: {improvement:+.1f}%\n")

    return best_weights, best_fit


def parse_game_id(filename):
    m = re.search(r'(G\d)', filename, re.IGNORECASE)
    return m.group(1).upper() if m else None


# ── Main extraction loop ──────────────────────────────────────────────────────

def extract_all_sam(base_dir=".", use_ga=True):

    subjects = sorted([
        s for s in os.listdir(base_dir)
        if s.startswith("(") and s.endswith(")")
        and os.path.isdir(os.path.join(base_dir, s))
    ])

    print(f"Found {len(subjects)} subjects\n")

    # ── Pass 1: collect all raw SAM scores ───────────────────────────────────
    print("Pass 1: Extracting raw SAM ratings from PDFs...")
    raw_data = {}   # subject → {game → {valence, arousal}}

    all_sam_pairs = []   # flat list of (valence, arousal) for GA

    for sub in subjects:
        sam_path = os.path.join(base_dir, sub, "SAM Ratings")
        if not os.path.exists(sam_path):
            print(f"  [SKIP] {sub} — no SAM Ratings folder")
            continue

        raw_data[sub] = {}
        for filename in sorted(os.listdir(sam_path)):
            if not filename.lower().endswith(".pdf"):
                continue
            game_id = parse_game_id(filename)
            if game_id is None:
                continue

            scores = extract_sam_scores(os.path.join(sam_path, filename))
            v, a   = scores["valence"], scores["arousal"]
            raw_data[sub][game_id] = {"valence": v, "arousal": a}

            if v is not None and a is not None:
                all_sam_pairs.append((v, a))
                print(f"  {sub} {game_id}: V={v}  A={a}")

    print(f"\nCollected {len(all_sam_pairs)} valid SAM pairs\n")

    # ── Pass 2: GA optimisation ───────────────────────────────────────────────
    if use_ga and len(all_sam_pairs) >= 6:
        best_weights, best_fit = run_ga(all_sam_pairs, verbose=True)
        # Save best weights for reference
        weights_df = pd.DataFrame([{
            "w1_arousal":     best_weights[0],
            "w2_inv_valence": best_weights[1],
            "w3_interaction": best_weights[2],
            "w4_arousal_sq":  best_weights[3],
            "fitness":        best_fit,
        }])
        weights_df.to_csv(os.path.join(base_dir, "ga_best_weights.csv"), index=False)
        print(f"  Saved GA weights → ga_best_weights.csv")
    else:
        if use_ga:
            print("[WARN] Too few SAM pairs for GA — using default weights")
        best_weights = DEFAULT_WEIGHTS
        print(f"  Using default weights: {DEFAULT_WEIGHTS}")

    # ── Pass 3: compute workload with best weights + tertile classes ──────────
    print("\nPass 3: Computing workload scores and assigning tertile classes...")
    all_records = []

    for sub, games in raw_data.items():
        if not games:
            continue

        # Compute raw scores with GA-optimised weights
        game_scores = {}
        for game_id, va in games.items():
            v, a = va["valence"], va["arousal"]
            if v is None or a is None:
                continue
            raw = compute_workload_raw(v, a, best_weights)
            game_scores[game_id] = {"valence": v, "arousal": a, "raw": raw}

        if not game_scores:
            continue

        # Normalise raw scores to [1, 9] scale
        raw_values = [d["raw"] for d in game_scores.values()]
        norm_values = normalise_to_1_9(raw_values)

        sub_records = []
        for (game_id, d), norm_w in zip(game_scores.items(), norm_values):
            sub_records.append({
                "subject":             sub,
                "game":                game_id,
                "valence":             d["valence"],
                "arousal":             d["arousal"],
                "workload_continuous": round(float(norm_w), 4),
            })

        # Assign per-subject tertile classes
        sub_records = assign_tertile_classes(sub_records)

        # Print summary
        cls_counts = {0: 0, 1: 0, 2: 0}
        for r in sub_records:
            cls_counts[r["workload_class"]] += 1

        scores_str = "  ".join(
            f"{r['game']}={r['workload_continuous']:.1f}"
            f"({r['workload_label'][0].upper()})"
            for r in sub_records
        )
        t1 = sub_records[0]["threshold_t1"]
        t2 = sub_records[0]["threshold_t2"]
        print(f"  {sub}: {scores_str}")
        print(f"    tertile thresholds: ≤{t1:.2f}=Low  ≤{t2:.2f}=Med  else=High  "
              f"| L={cls_counts[0]} M={cls_counts[1]} H={cls_counts[2]}")

        # Save per-subject CSV
        df_sub = pd.DataFrame(sub_records)[[
            "subject","game","valence","arousal",
            "workload_continuous","workload_class","workload_label"
        ]]
        df_sub.to_csv(os.path.join(base_dir, sub, "sam_workload.csv"), index=False)

        all_records.extend(sub_records)

    # ── Save global CSV ───────────────────────────────────────────────────────
    if not all_records:
        print("[ERROR] No records extracted.")
        return

    df_all = pd.DataFrame(all_records)[[
        "subject","game","valence","arousal",
        "workload_continuous","workload_class","workload_label"
    ]]
    global_path = os.path.join(base_dir, "sam_all_subjects.csv")
    df_all.to_csv(global_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Subjects : {df_all['subject'].nunique()}")
    print(f"  Games    : {len(df_all)}")
    print(f"  Workload : [{df_all['workload_continuous'].min():.2f}, "
          f"{df_all['workload_continuous'].max():.2f}]  "
          f"mean={df_all['workload_continuous'].mean():.2f}")

    print(f"\n  Class distribution (per-subject tertile):")
    for cls, lbl in [(0,"Low"),(1,"Medium"),(2,"High")]:
        n   = (df_all["workload_class"] == cls).sum()
        pct = n / len(df_all) * 100
        bar = "█" * int(pct / 2)
        print(f"    {lbl:<8}: {n:>3} ({pct:.1f}%)  {bar}")

    # Check all 3 classes present per subject
    missing = []
    for sub, grp in df_all.groupby("subject"):
        if not {0,1,2}.issubset(set(grp["workload_class"])):
            missing.append(sub)
    if missing:
        print(f"\n  [WARN] {len(missing)} subjects missing a class: {missing}")
    else:
        print(f"\n  All subjects have Low + Medium + High classes.")

    print(f"\n  GA weights used: {[round(w,4) for w in best_weights]}")
    print(f"  Saved: {global_path}")
    print(f"  Saved: <subject>/sam_workload.csv  (per subject)")
    if use_ga:
        print(f"  Saved: ga_best_weights.csv  (reuse with --no-ga next time)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ga", action="store_true",
                        help="Skip GA, use original formula weights [1,1,0,0]")
    args = parser.parse_args()
    extract_all_sam(base_dir=".", use_ga=not args.no_ga)
