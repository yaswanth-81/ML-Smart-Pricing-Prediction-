import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# ==========================================================
# CONFIGURATION
# ==========================================================
CSV_PATH = r"Y:\ML_CHALLENGE_2025\ml-pricing-challenge\dataset\test.csv"
IMG_DIR  = r"Y:\ML_CHALLENGE_2025\ml-pricing-challenge\test_images"
os.makedirs(IMG_DIR, exist_ok=True)

MAX_THREADS = 200        # 🔹 200–300 is the sweet spot for stability + speed
TIMEOUT = 8              # Skip slow connections
RETRIES = 2              # Retry twice for unstable URLs
BATCH_SIZE = 2000        # Download in manageable chunks

# ==========================================================
# STEP 1: Load dataset
# ==========================================================
df = pd.read_csv(CSV_PATH)
df.dropna(subset=['image_link', 'sample_id'], inplace=True)

# Filter rows where the image has not been downloaded yet
df_to_download = df[~df['sample_id'].apply(lambda x: os.path.exists(os.path.join(IMG_DIR, f"{x}.jpg")))]
print(f"🧾 Images to download: {len(df_to_download)} / {len(df)}")

# ==========================================================
# STEP 2: Download & Save JPEG Images (High Quality, Skip Slow)
# ==========================================================
def download_and_save_jpg(row):
    jpg_path = os.path.join(IMG_DIR, f"{row['sample_id']}.jpg")
    url = row['image_link']

    for attempt in range(RETRIES):
        try:
            response = requests.get(url, timeout=TIMEOUT)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert('RGB')
                img.save(jpg_path, format='JPEG', quality=95, subsampling=0)
                return True
            else:
                return False
        except Exception:
            time.sleep(0.2)  # short delay before retry
            continue
    return False  # ❌ failed after retries

# ==========================================================
# STEP 3: Parallel Batch Downloads
# ==========================================================
total_success, total_failed = 0, 0
rows = df_to_download.to_dict('records')

for i in range(0, len(rows), BATCH_SIZE):
    batch = rows[i:i+BATCH_SIZE]
    print(f"🚀 Processing batch {i//BATCH_SIZE + 1}/{(len(rows)//BATCH_SIZE)+1} "
          f"({len(batch)} images)...")

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {executor.submit(download_and_save_jpg, row): row['sample_id'] for row in batch}
        for future in tqdm(as_completed(futures), total=len(futures), desc="📸 Downloading", leave=False):
            try:
                if future.result():
                    total_success += 1
                else:
                    total_failed += 1
            except Exception:
                total_failed += 1

    print(f"✅ Batch done — success: {total_success}, failed/skipped: {total_failed}")

print(f"\n🎯 Final Report → Downloaded: {total_success} | Failed: {total_failed}")
print(f"📂 Images saved in: {IMG_DIR}")
