"""Download ASTROMER weights manually."""
import os
import sys
import requests
import zipfile

# Setup
weights_dir = r"C:\Users\alexy\Documents\Claude_projects\Kaggle competition\MALLORN astrophysics\weights"
os.makedirs(weights_dir, exist_ok=True)

# Download both weight files (macho_a0 and macho_a1)
# Based on ASTROMER GitHub, these are the actual weight files
urls = {
    "macho_a0": "https://github.com/astromer-science/weights/raw/main/macho_a0.zip",
    "macho_a1": "https://github.com/astromer-science/weights/raw/main/macho_a1.zip"
}

for name, url in urls.items():
    print(f"\nDownloading {name}...")
    target_dir = os.path.join(weights_dir, name)
    zip_path = f"{target_dir}.zip"

    try:
        # Download with requests
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        print(f"  Size: {total_size / 1024:.1f} KB")

        # Save to zip file
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"  Saved to: {zip_path}")

        # Verify it's a valid zip
        if zipfile.is_zipfile(zip_path):
            print(f"  Valid zip file!")

            # Extract
            os.makedirs(target_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            print(f"  Extracted to: {target_dir}")

            # List contents
            files = os.listdir(target_dir)
            print(f"  Contents: {files}")

            # Clean up zip
            os.remove(zip_path)
        else:
            print(f"  ERROR: Not a valid zip file!")
            # Check what we got
            with open(zip_path, 'rb') as f:
                header = f.read(100)
            print(f"  First bytes: {header[:50]}")

    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "="*50)
print("Downloaded weights:")
for item in os.listdir(weights_dir):
    item_path = os.path.join(weights_dir, item)
    if os.path.isdir(item_path):
        contents = os.listdir(item_path)
        print(f"  {item}/: {contents}")
