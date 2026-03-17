import os
import re
import pandas as pd
from pdfminer.high_level import extract_text


# ─── GAMEEMO SAM PDF Structure ────────────────────────────────────────────────
# Each SAM PDF contains two rating scales:
#   Scale 1 (Valence):  row of figures from unhappy(1) to happy(9)
#   Scale 2 (Arousal):  row of figures from calm(1) to excited(9)
# The participant circles ONE number per scale.
# In the extracted text, these appear as lone digits on their own lines.
# ──────────────────────────────────────────────────────────────────────────────

VALENCE_ANCHORS  = ["horrible", "unhappy", "sad", "annoyed", "unsatisfied",
                    "happy",    "funny",   "joyful", "satisfied", "pleased"]

AROUSAL_ANCHORS  = ["calm", "relaxed", "sleepy", "sluggish", "dull",
                    "excited", "stimulated", "aroused", "frenzied", "jittery"]

SAM_MIN = 1
SAM_MAX = 9


def extract_sam_scores(pdf_path: str) -> dict:
    """
    Extract valence and arousal scores from a GAMEEMO SAM PDF.

    Strategy (in priority order):
    1. Find standalone digits (1-9) near SAM anchor words → most reliable
    2. Fall back to extracting all standalone digits in document order
       (first = valence, second = arousal per GAMEEMO PDF layout)
    3. Return None for any score that cannot be determined
    """
    text = extract_text(pdf_path)
    lines = text.splitlines()

    # ── Strategy 1: anchor-proximity extraction ────────────────────────────
    valence_score = _find_score_near_anchors(lines, VALENCE_ANCHORS)
    arousal_score = _find_score_near_anchors(lines, AROUSAL_ANCHORS)

    # ── Strategy 2: positional fallback ───────────────────────────────────
    if valence_score is None or arousal_score is None:
        standalone_digits = _extract_standalone_digits(lines)

        if len(standalone_digits) >= 2:
            if valence_score is None:
                valence_score = standalone_digits[0]
            if arousal_score is None:
                arousal_score = standalone_digits[1]

        elif len(standalone_digits) == 1:
            # Only one digit found — log warning, keep what we have
            print(f"  [WARN] Only 1 digit found in {os.path.basename(pdf_path)}")

    # ── Validate range ─────────────────────────────────────────────────────
    valence_score = _validate_sam_range(valence_score, "valence", pdf_path)
    arousal_score = _validate_sam_range(arousal_score, "arousal", pdf_path)

    return {
        "valence": valence_score,   # 1=horrible, 9=happy
        "arousal": arousal_score,   # 1=calm,     9=excited
    }


def _find_score_near_anchors(lines: list, anchors: list) -> int | None:
    """
    Search lines for an anchor word, then look within ±3 lines for a
    standalone digit 1–9.
    """
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(anchor in line_lower for anchor in anchors):
            # Search window around this anchor line
            search_start = max(0, i - 3)
            search_end   = min(len(lines), i + 4)

            for j in range(search_start, search_end):
                match = re.match(r'^\s*([1-9])\s*$', lines[j])
                if match:
                    return int(match.group(1))

            # Also check inline: "Valence: 7" or "Rating: 3"
            inline = re.search(r'\b([1-9])\b', line)
            if inline:
                return int(inline.group(1))

    return None


def _extract_standalone_digits(lines: list) -> list[int]:
    """
    Extract all digits (1–9) that appear alone on a line.
    GAMEEMO SAM forms have the circled rating as a lone number.
    """
    digits = []
    for line in lines:
        match = re.match(r'^\s*([1-9])\s*$', line)
        if match:
            digits.append(int(match.group(1)))
    return digits


def _validate_sam_range(score, name: str, pdf_path: str) -> int | None:
    """Ensure score is within valid SAM range 1–9."""
    if score is None:
        print(f"  [WARN] Could not extract {name} from {os.path.basename(pdf_path)}")
        return None
    if not (SAM_MIN <= score <= SAM_MAX):
        print(f"  [WARN] {name}={score} out of range in {os.path.basename(pdf_path)}, discarding")
        return None
    return score


def compute_workload(valence: int | None,
                     arousal: int | None) -> dict | None:
    """
    Compute cognitive workload proxy from SAM valence and arousal.

    Scientific basis:
        Russell's Circumplex Model: workload occupies the high-arousal,
        low-valence quadrant (stressed/tense). This is consistent with
        DEAP, MAHNOB-HCI, and affective gaming BCI literature.

    Formula:
        workload_raw  = arousal + (10 - valence)    ∈ [2, 18]
        workload_norm = ((raw - 2) / 16) × 8 + 1   ∈ [1, 9]

    Classification bins (equal-width on 1–9 scale):
        Low    → [1.00, 3.67)
        Medium → [3.67, 6.33)
        High   → [6.33, 9.00]

    Returns dict with continuous score + discrete class label, or None
    if either input is missing.
    """
    if valence is None or arousal is None:
        return None

    # ── Continuous workload score ──────────────────────────────────────────
    raw        = arousal + (10 - valence)           # range: 2–18
    normalized = ((raw - 2) / 16) * 8 + 1           # rescaled: 1–9

    # ── 3-class label ─────────────────────────────────────────────────────
    if normalized < (1 + 8/3):        # < 3.67
        workload_class = 0            # Low
        class_label    = "low"
    elif normalized < (1 + 16/3):     # < 6.33
        workload_class = 1            # Medium
        class_label    = "medium"
    else:
        workload_class = 2            # High
        class_label    = "high"

    return {
        "workload_continuous": round(normalized, 4),
        "workload_class":      workload_class,       # 0/1/2
        "workload_label":      class_label,          # low/medium/high
        "arousal_raw":         arousal,
        "valence_raw":         valence,
    }


def parse_game_id(filename: str) -> str | None:
    """Extract game ID (G1–G5) from SAM PDF filename."""
    match = re.search(r'(G\d)', filename, re.IGNORECASE)
    return match.group(1).upper() if match else None


# ─── Main extraction loop ─────────────────────────────────────────────────────

def extract_all_sam(base_dir: str = ".") -> None:

    subjects = sorted([
        s for s in os.listdir(base_dir)
        if s.startswith("(") and s.endswith(")")
        and os.path.isdir(os.path.join(base_dir, s))
    ])

    print(f"Found {len(subjects)} subjects: {subjects}\n")

    all_subjects_summary = []

    for sub in subjects:

        sam_path = os.path.join(base_dir, sub, "SAM Ratings")

        if not os.path.exists(sam_path):
            print(f"[SKIP] {sub} — no SAM Ratings folder found")
            continue

        rows = []
        print(f"Processing {sub}...")

        for filename in sorted(os.listdir(sam_path)):

            if not filename.lower().endswith(".pdf"):
                continue

            game_id = parse_game_id(filename)
            if game_id is None:
                print(f"  [SKIP] Cannot parse game ID from: {filename}")
                continue

            pdf_path = os.path.join(sam_path, filename)
            scores   = extract_sam_scores(pdf_path)
            result   = compute_workload(scores["valence"], scores["arousal"])

            if result is None:
                print(f"  [FAIL] {filename} — incomplete scores: {scores}")
                row = {
                    "subject":             sub,
                    "game":                game_id,
                    "valence":             scores["valence"],
                    "arousal":             scores["arousal"],
                    "workload_continuous": None,
                    "workload_class":      None,
                    "workload_label":      None,
                }
            else:
                print(f"  {game_id}: V={result['valence_raw']} A={result['arousal_raw']}"
                      f" → workload={result['workload_continuous']:.2f} ({result['workload_label']})")
                row = {
                    "subject":             sub,
                    "game":                game_id,
                    "valence":             result["valence_raw"],
                    "arousal":             result["arousal_raw"],
                    "workload_continuous": result["workload_continuous"],
                    "workload_class":      result["workload_class"],
                    "workload_label":      result["workload_label"],
                }

            rows.append(row)
            all_subjects_summary.append(row)

        if not rows:
            print(f"  [WARN] No valid SAM data extracted for {sub}")
            continue

        df = pd.DataFrame(rows)

        # ── Save per-subject CSV ───────────────────────────────────────────
        out_path = os.path.join(base_dir, sub, "sam_workload.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved → {out_path}")

        # ── Per-subject sanity check ───────────────────────────────────────
        valid = df.dropna(subset=["workload_continuous"])
        print(f"  Valid games: {len(valid)}/{len(df)} | "
              f"Workload range: [{valid['workload_continuous'].min():.2f}, "
              f"{valid['workload_continuous'].max():.2f}] | "
              f"Class dist: {valid['workload_class'].value_counts().to_dict()}\n")

    # ── Save global CSV across all subjects ───────────────────────────────────
    global_df = pd.DataFrame(all_subjects_summary)
    global_path = os.path.join(base_dir, "sam_all_subjects.csv")
    global_df.to_csv(global_path, index=False)

    print(f"\n{'='*60}")
    print(f"Global summary saved → {global_path}")
    print(f"Total entries: {len(global_df)} | "
          f"Valid: {global_df['workload_continuous'].notna().sum()}")
    print(f"Overall class distribution:\n"
          f"{global_df['workload_label'].value_counts().to_string()}")


if __name__ == "__main__":
    extract_all_sam(base_dir=".")