from pdfminer.high_level import extract_text

text = extract_text("(S02)/SAM Ratings/G3.pdf")
print(text)
