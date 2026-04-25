import os
import requests

def download_from_gdrive(file_id, destination):
    print(f"[…] Downloading {destination}...")
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)

    # Handle Google's virus scan warning for large files
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

    print(f"[✓] Saved {destination}")


def download_datasets():
    files = {
        "tmdb_5000_movies.csv":  "1EZTGme9vrzeuB_HKU1eBLgInCXWb-dRx",
        "tmdb_5000_credits.csv": "1JYAFQHX6IYQB83HwkVKtgZG6aeZCqIhF",
    }

    all_present = all(os.path.exists(f) for f in files)
    if all_present:
        print("[✓] CSV files already present.")
        return

    print("[…] CSV files not found. Downloading from Google Drive...")
    for filename, file_id in files.items():
        if not os.path.exists(filename):
            download_from_gdrive(file_id, filename)

    print("[✓] All CSV files downloaded successfully!")


if __name__ == "__main__":
    download_datasets()