import re
from pdfminer.high_level import extract_text

def extract_boredom_score(pdf_path):
    text = extract_text(pdf_path)

    pattern = r"boring.*?\n\s*(\d+)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

    if match:
        boredom = int(match.group(1))
        return boredom
    else:
        return None


# TEST ON ONE FILE
pdf_path = "(S03)/SAM Ratings/G1.pdf"

boredom_score = extract_boredom_score(pdf_path)

print("Boredom Score:", boredom_score)

if boredom_score is not None:
    workload = 11 - boredom_score
    print("Computed Workload:", workload)
