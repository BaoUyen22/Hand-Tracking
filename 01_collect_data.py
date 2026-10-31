from src.utils import normalize_hand_landmarks_84d

SAVE_DIR = "data/processed"
os.makedirs(SAVE_DIR, exist_ok=True)
np.save("data/processed/X.npy", np.array(all_features, dtype=np.float32))
np.save("data/processed/y.npy", np.array(all_labels, dtype=np.int64))
np.save("data/processed/classes.npy", np.array(LABELS))