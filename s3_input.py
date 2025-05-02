import os
import zipfile

from s3fs import S3FileSystem

# 1) Read AIchor’s S3 API URL and your input‐bucket prefix
endpoint = os.environ["S3_ENDPOINT"]
input_prefix = os.environ["AICHOR_INPUT_PATH"].rstrip(
    "/"
)  # e.g. "tinkerer-8289ccc77ccb4ed0-inputs" :contentReference[oaicite:0]{index=0}

# 2) Initialize the S3 client
s3 = S3FileSystem(client_kwargs={"endpoint_url": endpoint})

# 3) Build the full key to your zip file
remote_zip_key = f"{input_prefix}/Aide_datsets/Datasets.zip"

# 4) Download it locally
local_zip = "Datasets.zip"
with s3.open(remote_zip_key, "rb") as src, open(local_zip, "wb") as dst:
    dst.write(src.read())

print(f"Downloaded s3://{remote_zip_key} → {local_zip}")

# 5) Unzip into a folder (e.g. ./data/)
extract_dir = "./Datasets"
os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(local_zip, "r") as z:
    z.extractall(extract_dir)
print(f"Extracted all files to {extract_dir}")
