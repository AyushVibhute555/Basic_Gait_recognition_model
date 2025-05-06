import os
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Paths
FEATURE_FILE = 'features.csv'
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'gait_model.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

# Load gait features from CSV
print("[INFO] Loading features...")
df = pd.read_csv(FEATURE_FILE, header=None)
labels = df.iloc[:, 0]
features = df.iloc[:, 1:]

# Encode string labels to integers
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

# Train an SVM classifier
print("[INFO] Training SVM model...")
clf = SVC(kernel='linear', probability=True)
clf.fit(features, encoded_labels)

# Save the model and label encoder
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(clf, MODEL_PATH)
joblib.dump(le, LABEL_ENCODER_PATH)

print(f"[✅] Model saved to: {MODEL_PATH}")
print(f"[✅] Label encoder saved to: {LABEL_ENCODER_PATH}")
