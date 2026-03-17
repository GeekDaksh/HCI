import numpy as np
import os


def load_dataset():

    path = "processed/dataset.npz"

    if not os.path.exists(path):
        raise FileNotFoundError("Run aggregate_sessions.py first")

    data = np.load(path, allow_pickle=True)

    X = data["X"].astype(np.float32)
    y_class = data["y_class"].astype(np.int32)
    y_cont = data["y_cont"].astype(np.float32)

    subject_ids = data["subject_ids"].astype(np.int32)
    subjects = data["subjects"]
    games = data["games"]

    return X, y_class, y_cont, subject_ids, subjects, games