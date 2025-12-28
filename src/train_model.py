import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = r"Y:\ML_CHALLENGE_2025\ml-pricing-challenge\features"
MODEL_PATH = os.path.join(DATA_DIR, "hybrid_model_trained.h5")

# ============================================================
# LOAD PREPROCESSED DATA
# ============================================================
print("📂 Loading preprocessed data...")

train_df = pd.read_csv(os.path.join(DATA_DIR, "train_features.csv"))
img_features = np.load(os.path.join(DATA_DIR, "train_img_features.npy"))
img_ids = pd.read_csv(os.path.join(DATA_DIR, "train_img_ids.csv"))
X_text = joblib.load(os.path.join(DATA_DIR, "X_train_tfidf.pkl"))

# Merge to align features
df = train_df.merge(img_ids, on="sample_id", how="inner")
print(f"✅ Loaded {len(df)} aligned training samples.")

# Ensure correspondence
X_img = img_features[:len(df)]
y = df["price"].values  # adjust column name if different

print(f"🧩 Image features: {X_img.shape}, Text features: {X_text.shape}, Labels: {y.shape}")

# ============================================================
# SPLIT TRAIN / VALIDATION
# ============================================================
X_img_train, X_img_val, X_text_train, X_text_val, y_train, y_val = train_test_split(
    X_img, X_text, y, test_size=0.2, random_state=42
)

# ============================================================
# HYBRID MODEL DEFINITION
# ============================================================
print("⚙️ Building hybrid model...")

# Image input branch
inp_img = Input(shape=(X_img_train.shape[1],), name="image_input")
x1 = Dense(512, activation="relu")(inp_img)
x1 = Dropout(0.3)(x1)

# Text input branch
inp_txt = Input(shape=(X_text_train.shape[1],), name="text_input")
x2 = Dense(512, activation="relu")(inp_txt)
x2 = Dropout(0.3)(x2)

# Combine both
combined = concatenate([x1, x2])
x = Dense(256, activation="relu")(combined)
x = Dropout(0.3)(x)
output = Dense(1, activation="linear")(x)  # for regression

model = Model(inputs=[inp_img, inp_txt], outputs=output)
model.compile(optimizer=Adam(1e-4), loss="mae", metrics=["mse"])
model.summary()

# ============================================================
# TRAINING
# ============================================================
checkpoint = ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

print("🚀 Training model...")
history = model.fit(
    [X_img_train, X_text_train],
    y_train,
    validation_data=([X_img_val, X_text_val], y_val),
    epochs=25,
    batch_size=64,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

print(f"✅ Training complete. Model saved to {MODEL_PATH}")

# ============================================================
# EVALUATION
# ============================================================
val_loss, val_mse = model.evaluate([X_img_val, X_text_val], y_val, verbose=0)
print(f"🎯 Validation MAE: {val_loss:.4f} | MSE: {val_mse:.4f}")
