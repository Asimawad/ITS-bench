import os
from datetime import datetime
from pathlib import Path

from s3fs import S3FileSystem
from utils_zip import make_filtered_zip

if __name__ == "__main__":
    # --- Step 1: Get environment variables ---
    endpoint = os.environ["S3_ENDPOINT"]
    output_prefix = os.environ["AICHOR_OUTPUT_PATH"].rstrip("/")

    # --- Step 2: Set up output paths ---
    local_folder = "runs"
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    zip_base_name = f"runs_{timestamp}"
    local_zip_path = f"./{zip_base_name}.zip"  # make_filtered_zip will ensure the .zip suffix
    remote_zip_key = f"{output_prefix}/{zip_base_name}.zip"

    # --- Step 3: Zip the runs/ directory using make_filtered_zip ---
    print(f"[INFO] Using make_filtered_zip to archive: {local_folder} → {local_zip_path}")

    if not os.path.isdir(local_folder):
        print(f"[ERROR] Source folder '{local_folder}' not found. Nothing to zip.")
    else:
        # Call make_filtered_zip to create the archive with built-in exclusions and file existence checks
        result_path = make_filtered_zip(zip_base_name, local_folder)

        # Convert Path object to string for S3 upload if needed
        actual_local_zip_path = str(result_path)

        # --- Step 4: Initialize S3 and upload zip ---
        print(f"[INFO] Uploading to S3: s3://{remote_zip_key}")
        s3 = S3FileSystem(client_kwargs={"endpoint_url": endpoint})

        with open(actual_local_zip_path, "rb") as f_local:
            with s3.open(remote_zip_key, "wb") as f_remote:
                f_remote.write(f_local.read())

        print(f"[INFO] Successfully uploaded {zip_base_name}.zip to s3://{remote_zip_key}")

    # --- (Optional) Upload the full folder tree [DISABLED] ---
    # print("[INFO] Uploading full runs/ directory...")
    # for root, _, files in os.walk(local_folder):
    #     for file in files:
    #         local_path = os.path.join(root, file)
    #         rel_path = os.path.relpath(local_path, local_folder)
    #         s3_path = f"{output_prefix}/runs/{rel_path}"
    #         with open(local_path, "rb") as f_local:
    #             with s3.open(s3_path, "wb") as f_remote:
    #                 f_remote.write(f_local.read())
    # print(f"[INFO] Full directory uploaded to s3://{output_prefix}/runs/")
