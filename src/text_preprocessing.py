import re
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm

# ======================================================
# CONFIG
# ======================================================
TRAIN_CSV = r"Y:\ML_CHALLENGE_2025\ml-pricing-challenge\dataset\train.csv"
TEST_CSV  = r"Y:\ML_CHALLENGE_2025\ml-pricing-challenge\dataset\test.csv"
OUTPUT_DIR = r"Y:\ML_CHALLENGE_2025\ml-pricing-challenge\features"
N_FEATURES = 5000  # adjust based on RAM
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# HELPER FUNCTIONS
# ======================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_ipq(text):
    """Extract Item Pack Quantity (e.g., 'Pack of 3', '2L', '5 pcs')"""
    text = str(text).lower()
    ipq_match = re.search(r'(\d+)\s?(pack|pcs|pieces|x|unit|litre|kg|ml|l)\b', text)
    return int(ipq_match.group(1)) if ipq_match else 1

def extract_brand(text):
    """Assume brand is often first capitalized word before product name"""
    text = str(text)
    match = re.match(r'([A-Za-z0-9&\-\']+)', text.strip())
    return match.group(1).lower() if match else "unknown"

# ======================================================
# LOAD DATA
# ======================================================
print("📂 Loading data...")
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

for df in [train_df, test_df]:
    df['catalog_content'] = df['catalog_content'].fillna("").astype(str)
    df['clean_text'] = df['catalog_content'].apply(clean_text)
    df['ipq'] = df['catalog_content'].apply(extract_ipq)
    df['brand'] = df['catalog_content'].apply(extract_brand)

# ======================================================
# ENCODE BRAND
# ======================================================
print("🔠 Encoding brand names...")
le = LabelEncoder()
train_df['brand_enc'] = le.fit_transform(train_df['brand'])
test_df['brand_enc'] = le.transform(test_df['brand'].map(lambda b: b if b in le.classes_ else "unknown"))

# ======================================================
# TF-IDF FEATURES
# ======================================================
print("🔤 Generating TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=N_FEATURES,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8,
    stop_words='english'
)
X_train_tfidf = tfidf.fit_transform(train_df['clean_text'])
X_test_tfidf = tfidf.transform(test_df['clean_text'])

# ======================================================
# SAVE OUTPUTS
# ======================================================
print("💾 Saving processed data...")
train_df[['sample_id', 'ipq', 'brand_enc', 'price']].to_csv(f"{OUTPUT_DIR}/train_features.csv", index=False)
test_df[['sample_id', 'ipq', 'brand_enc']].to_csv(f"{OUTPUT_DIR}/test_features.csv", index=False)

joblib.dump(X_train_tfidf, f"{OUTPUT_DIR}/X_train_tfidf.pkl")
joblib.dump(X_test_tfidf, f"{OUTPUT_DIR}/X_test_tfidf.pkl")
joblib.dump(tfidf, f"{OUTPUT_DIR}/tfidf_vectorizer.pkl")

print("✅ Text preprocessing complete! Saved TF-IDF features & metadata.")
