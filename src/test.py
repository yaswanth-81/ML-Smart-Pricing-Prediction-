import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanAbsoluteError  # for custom_objects

# ============================================================
# CONFIGURATION
# ============================================================
FEATURE_DIR = r"Y:\ML_CHALLENGE_2025\ml-pricing-challenge\features_combined"  # directory where features are stored
MODEL_PATH = r"Y:\ML_CHALLENGE_2025\ml-pricing-challenge\features\hybrid_model_trained.h5"

# ============================================================
# 1️⃣ Load Precomputed Image Features
# ============================================================
print("📂 Loading precomputed image features...")
image_features_path = os.path.join(FEATURE_DIR, "image_features.npy")
if not os.path.exists(image_features_path):
    raise FileNotFoundError(f"❌ {image_features_path} not found!")
image_features = np.load(image_features_path)
print(f"✅ Image features loaded: {image_features.shape}")

# ============================================================
# 2️⃣ Load Metadata
# ============================================================
metadata_path = os.path.join(FEATURE_DIR, "metadata.csv")
if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"❌ {metadata_path} not found!")

metadata = pd.read_csv(metadata_path)
print(f"✅ Metadata loaded: {metadata.shape}")

# ============================================================
# 3️⃣ Load TF-IDF Vectorizer & Text Features
# ============================================================
tfidf_vectorizer_path = os.path.join(FEATURE_DIR, "tfidf_vectorizer.pkl")
text_tfidf_path = os.path.join(FEATURE_DIR, "text_tfidf.pkl")

if os.path.exists(text_tfidf_path):
    print("📂 Loading precomputed text TF-IDF features...")
    text_features = joblib.load(text_tfidf_path)
    print(f"✅ Text features loaded: {text_features.shape}")
else:
    if not os.path.exists(tfidf_vectorizer_path):
        raise FileNotFoundError(f"❌ {tfidf_vectorizer_path} not found!")
    print("📂 Loading TF-IDF vectorizer...")
    vectorizer = joblib.load(tfidf_vectorizer_path)

    # Try to find a text column automatically
    text_col = None
    for col in metadata.columns:
        if metadata[col].dtype == "object" and metadata[col].str.len().mean() > 5:
            text_col = col
            break

    if text_col is None:
        raise ValueError("❌ No suitable text column found in metadata.csv! Please check the file.")
    
    print(f"📝 Using text column: {text_col}")
    text_features = vectorizer.transform(metadata[text_col].fillna(""))
    print(f"✅ Text features generated: {text_features.shape}")

# ============================================================
# Convert sparse to dense if needed
# ============================================================
if not isinstance(text_features, np.ndarray):
    text_features = text_features.toarray()  # csr_matrix → dense
    print(f"✅ Text features converted to dense: {text_features.shape}")

# ============================================================
# 4️⃣ Load Sample IDs
# ============================================================
image_ids_path = os.path.join(FEATURE_DIR, "image_ids.csv")
if os.path.exists(image_ids_path):
    ids_df = pd.read_csv(image_ids_path)
    sample_ids = ids_df.iloc[:, 0].values  # assumes first column is ID
    print(f"✅ Sample IDs loaded: {len(sample_ids)}")
else:
    sample_ids = np.arange(len(metadata))
    print(f"⚠️ image_ids.csv not found, using default IDs: {len(sample_ids)}")

# ============================================================
# 5️⃣ Load the Trained Model (with custom_objects)
# ============================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ {MODEL_PATH} not found!")

print("🧠 Loading trained hybrid model...")
model = load_model(MODEL_PATH, custom_objects={"mae": MeanAbsoluteError()})
print("✅ Model loaded successfully.")

# ============================================================
# 6️⃣ Generate Predictions
# ============================================================
print("🔮 Generating predictions...")

# Match model input dimensions safely
expected_dim = model.input[1].shape[1]
if text_features.shape[1] != expected_dim:
    print(f"⚠️ Adjusting text feature dimensions from {text_features.shape[1]} → {expected_dim}")
    new_text_features = np.zeros((text_features.shape[0], expected_dim))
    new_text_features[:, :text_features.shape[1]] = text_features
    text_features = new_text_features

predictions = model.predict([image_features, text_features], verbose=1)
predictions = predictions.flatten()
print(f"✅ Predictions generated: {predictions.shape}")

# ============================================================
# 7️⃣ Save Submission
# ============================================================
submission_path = r"Y:\ML_CHALLENGE_2025\ml-pricing-challenge\outputs\submission.csv"
submission = pd.DataFrame({
    "id": sample_ids,
    "price": predictions
})
submission.to_csv(submission_path, index=False)
print(f"📁 Submission saved as {submission_path} ✅")
