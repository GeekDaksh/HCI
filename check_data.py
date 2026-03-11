import os
import numpy as np

print("Current directory:", os.getcwd())

files = os.listdir("processed")
print("Processed files:", files)

# Load the first processed subject automatically
data = np.load(os.path.join("processed", files[0]))

print("X shape:", data["X"].shape)
print("y shape:", data["y"].shape)
print("Workload value:", np.unique(data["y"]))
print("Workload range:", data["y"].min(), "to", data["y"].max())

