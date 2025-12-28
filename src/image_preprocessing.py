import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras import Input
import tensorflow as tf

# ============================================================
# CONFIGURATION
# ============================================================
TRAIN_IMG_DIR = r"Y:\ML_CHALLENGE_2025\ml-pricing-challenge\images_jpg"
OUTPUT_DIR = r"Y:\ML_CHALLENGE_2025\ml-pricing-challenge\features"
IMG_SIZE = (224, 224)
LOAD_BATCH = 2000       # Load this many images into memory
PRED_BATCH = 64         # Model inference batch size
NUM_THREADS = 8         # Parallel image loading
SAVE_INTERVAL = 2000    # Save progress every N images

os.makedirs(OUTPUT_DIR, exist_ok=True)
feature_path = os.path.join(OUTPUT_DIR, "test_img_features.npy")
ids_path = os.path.join(OUTPUT_DIR, "test_img_ids.csv")

# ============================================================
# LOAD MODEL (RGB SAFE)
# ============================================================
print("📦 Initializing EfficientNetB0 (RGB)...")
input_tensor = Input(shape=(224, 224, 3))
base_model = EfficientNetB0(include_top=False, pooling='avg', weights=None, input_tensor=input_tensor)
weights_url = "https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5"
weights_path = tf.keras.utils.get_file("efficientnetb0_notop.h5", weights_url)
base_model.load_weights(weights_path)
model = Model(inputs=base_model.input, outputs=base_model.output)
print("✅ Model ready with pretrained ImageNet weights.\n")

# ============================================================
# LOAD FILE LIST & CHECK FOR RESUME
# ============================================================
all_files = [f for f in os.listdir(TRAIN_IMG_DIR) if f.lower().endswith(".jpg")]
processed_ids = []

if os.path.exists(ids_path):
    processed_ids = pd.read_csv(ids_path)["sample_id"].astype(str).tolist()

remaining_files = [
    f for f in all_files if os.path.splitext(f)[0] not in processed_ids
]

print(f"📂 Total images: {len(all_files)} | Already done: {len(processed_ids)} | Remaining: {len(remaining_files)}\n")

# ============================================================
# IMAGE PREPROCESS FUNCTION
# ============================================================
def load_and_preprocess(img_file):
    try:
        img_path = os.path.join(TRAIN_IMG_DIR, img_file)
        img = image.load_img(img_path, target_size=IMG_SIZE)
        x = image.img_to_array(img)
        if x.shape[2] == 1:
            x = np.repeat(x, 3, axis=2)
        x = preprocess_input(x)
        return os.path.splitext(img_file)[0], x
    except Exception:
        return None, None

# ============================================================
# STREAMED FEATURE EXTRACTION
# ============================================================
if not remaining_files:
    print("✅ All images already processed. Nothing to do!")
else:
    total = len(remaining_files)
    processed_so_far = len(processed_ids)
    features_buffer = []
    ids_buffer = []

    print("🚀 Starting incremental feature extraction...\n")

    for start in range(0, total, LOAD_BATCH):
        end = min(start + LOAD_BATCH, total)
        batch_files = remaining_files[start:end]

        print(f"\n⚙️ Loading images {start+1 + processed_so_far}–{end + processed_so_far} / {len(all_files)}")

        # Load in parallel
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            results = list(tqdm(executor.map(load_and_preprocess, batch_files), total=len(batch_files), desc="🖼️ Loading"))

        valid_results = [(sid, img) for sid, img in results if sid is not None]
        if not valid_results:
            continue

        valid_ids, imgs = zip(*valid_results)
        imgs = np.stack(imgs, axis=0)

        # Extract features in batches
        feats = []
        for i in tqdm(range(0, len(imgs), PRED_BATCH), desc="🔍 Extracting"):
            feat_batch = model.predict(imgs[i:i + PRED_BATCH], verbose=0)
            feats.append(feat_batch)
        feats = np.vstack(feats)

        # Add to buffers
        features_buffer.append(feats)
        ids_buffer.extend(valid_ids)

        # Save intermediate progress every few thousand images
        if (start + LOAD_BATCH >= total) or ((start + LOAD_BATCH) % SAVE_INTERVAL == 0):
            print("💾 Saving progress...")
            all_features = (
                np.load(feature_path)
                if os.path.exists(feature_path) else np.empty((0, feats.shape[1]))
            )
            combined_features = np.vstack([all_features] + features_buffer)
            np.save(feature_path, combined_features)
            pd.DataFrame({'sample_id': processed_ids + ids_buffer}).to_csv(ids_path, index=False)

            print(f"✅ Saved up to {len(processed_ids) + len(ids_buffer)} images.")
            features_buffer.clear()  # Free memory

        # Cleanup
        del imgs, feats, results, valid_results
        tf.keras.backend.clear_session()

    print("\n🎯 All batches processed successfully!")

print("\n✅ FINAL SUMMARY:")
final_feats = np.load(feature_path)
final_ids = pd.read_csv(ids_path)
print(f"🧩 Total features saved: {final_feats.shape[0]} | Dim: {final_feats.shape[1]}")
print(f"📁 Stored in: {OUTPUT_DIR}")
