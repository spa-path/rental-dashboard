# data_fetcher.py
# Downloads Zillow CSVs if they don’t already exist locally.

import os
import requests
import urllib.request

# Zillow public URLs
REQUIRED_FILES = {
    "zillow_home_values_2025-07-03.csv": "https://files.zillowstatic.com/research/public/zhvi/Zip_ZHVI_AllHomes.csv",
    "zillow_rent_index_2025-07-03.csv": "https://files.zillowstatic.com/research/public/observed-rent-index/Zip_Observed_Rent_Index_AllHomes.csv"
}


def fetch_if_missing(data_dir="data"):
    abs_data_path = os.path.abspath(data_dir)
    print("📦 [DEBUG] Expected write path:", abs_data_path)
    os.makedirs(data_dir, exist_ok=True)
    print("📦 Data directory absolute path:", os.path.abspath(data_dir))  # <-- Add this
    for filename, url in REQUIRED_FILES.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            print(f"[INFO] Downloading {filename}...")
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                with open(path, "wb") as f:
                    f.write(response.content)
                print(f"[INFO] Saved to {path}")
            except Exception as e:
                print(f"[ERROR] Could not download {filename}: {e}")
    print("✅ Final contents of data/:", os.listdir(data_dir))

    



def download_if_missing(url: str, save_path: str) -> bool:
    """
    Download file from `url` to `save_path` only if not already present.
    Returns True if downloaded, False if already existed.
    """
    if os.path.exists(save_path):
        return False

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

# Example usage:
if __name__ == "__main__":
    rent_url = "https://files.zillowstatic.com/research/public/observed-rent-index/Zip_Ordinance_ORI_AllHomes_SA_month.csv"
    home_url = "https://files.zillowstatic.com/research/public/zhvi/Zip_ZHVI_AllHomes.csv"

    rent_path = "data/zillow_rent_index_latest.csv"
    home_path = "data/zillow_home_values_latest.csv"

    dl1 = download_if_missing(rent_url, rent_path)
    dl2 = download_if_missing(home_url, home_path)

    print("Downloaded:", dl1 or dl2)
