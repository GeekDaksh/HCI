import os
import re
import pandas as pd
from pdfminer.high_level import extract_text

def extract_scores(pdf_path):
    text = extract_text(pdf_path)

    patterns = {
        "satisfaction": r"satisfied.*?\n\s*(\d+)",
        "boring": r"boring.*?\n\s*(\d+)",
        "horrible": r"horrible.*?\n\s*(\d+)",
        "calm": r"calm.*?\n\s*(\d+)",
        "funny": r"funny.*?\n\s*(\d+)"
    }

    scores = {}

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            scores[key] = int(match.group(1))
        else:
            scores[key] = None

    return scores


def compute_workload(scores):
    boredom = scores["boring"]
    calm = scores["calm"]
    horrible = scores["horrible"]

    if None in (boredom, calm, horrible):
        return None

    workload = ((11 - boredom) + (11 - calm) + horrible) / 3
    return round(workload, 2)


subjects = [s for s in os.listdir() if s.startswith("(") and s.endswith(")")]

for sub in subjects:
    sam_path = os.path.join(sub, "SAM Ratings")
    rows = []

    for file in sorted(os.listdir(sam_path)):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(sam_path, file)

            scores = extract_scores(pdf_path)
            workload = compute_workload(scores)

            rows.append({
                "game": file.replace(".pdf", ""),
                "satisfaction": scores["satisfaction"],
                "boring": scores["boring"],
                "horrible": scores["horrible"],
                "calm": scores["calm"],
                "funny": scores["funny"],
                "workload": workload
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(sub, "sam_workload.csv"), index=False)

    print(f"✅ SAM workload saved for {sub}")

print("ALL SAM FILES REBUILT")
