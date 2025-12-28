import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# ================= CONFIG =================
CSV_PATH = r"Y:\ML_CHALLENGE_2025\ml-pricing-challenge\dataset\sample_test.csv"
IMG_DIR = r"Y:\ML_CHALLENGE_2025\ml-pricing-challenge\images_jpg"

OUTPUT_TEXT_FEATURES = r"Y:\ML_CHALLENGE_2025\ml-pricing-challenge\features\X_sample_test_tfidf.pkl"
OUTPUT_IMG_FEATURES  = r"Y:\ML_CHALLENGE_2025\ml-pricing-challenge\features\sample_test_img_features.npy"
OUTPUT_IMG_IDS       = r"Y:\ML_CHALLENGE_2025\ml-pricing-challenge\features\sample_test_img_ids.csv"

TEXT_COLUMN = "title"  # Change if needed
IMG_SIZE = (224, 224)  # EfficientNetB0 default input

# ================= LOAD CSV =================
print("📥 Loading CSV...")
df = pd.read_csv(CSV_PATH)
sample_ids = df["sample_id"].astype(str).tolist()
texts = df[TEXT_COLUMN].astype(str).tolist()

# ================= TEXT FEATURE EXTRACTION =================
print("📝 Extracting TF-IDF text features...")
vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features
text_features = vectorizer.fit_transform(texts).toarray()  # shape = (num_samples, num_text_features)

with open(OUTPUT_TEXT_FEATURES, "wb") as f:
    pickle.dump(text_features, f)
print(f"✅ Text features saved to {OUTPUT_TEXT_FEATURES}")

# ================= IMAGE FEATURE EXTRACTION =================
print("🖼️ Extracting image features using EfficientNetB0...")

# Load EfficientNetB0 without top layer for feature extraction
base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
img_features_list = []
img_ids_list = []

for sid in tqdm(sample_ids):
    img_path = os.path.join(IMG_DIR, f"{sid}.jpg")  # assuming images named as sample_id.jpg
    if not os.path.exists(img_path):
        print(f"⚠️ Image not found: {img_path}, using zeros.")
        img_array = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3))
    else:
        img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
        img_array = np.array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feat = base_model.predict(img_array, verbose=0)
    img_features_list.append(feat.flatten())
    img_ids_list.append(sid)

img_features = np.stack(img_features_list)
np.save(OUTPUT_IMG_FEATURES, img_features)
pd.DataFrame({"sample_id": img_ids_list}).to_csv(OUTPUT_IMG_IDS, index=False)

print(f"✅ Image features saved to {OUTPUT_IMG_FEATURES}")
print(f"✅ Image IDs saved to {OUTPUT_IMG_IDS}")
