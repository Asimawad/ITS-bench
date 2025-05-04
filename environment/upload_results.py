# upload_result.py
import os
import shutil
from pathlib import Path

from s3fs import S3FileSystem


def upload_to_s3(zip_path, destination_key=None):
    """Uploads the given zip file to the AICHOR output S3 bucket."""
    endpoint = os.environ["S3_ENDPOINT"]
    output_prefix = os.environ["AICHOR_OUTPUT_PATH"].rstrip("/")

    # Build the key if not provided
    if destination_key is None:
        destination_key = f"{output_prefix}/{os.path.basename(zip_path)}"

    s3 = S3FileSystem(client_kwargs={"endpoint_url": endpoint})

    with open(zip_path, "rb") as src, s3.open(destination_key, "wb") as dst:
        dst.write(src.read())

    print(f"[upload_result] Uploaded {zip_path} to s3://{destination_key}")


def upload_to_local(zip_path, destination_key=None, *, make_outputs_dir=True):
    """
    Copy *zip_path* to *destination_key* on the local filesystem.
    If destination_key is omitted, the file is placed in ./outputs/.

    - If src and dst are identical, the copy is skipped.
    """
    src = Path(zip_path).resolve()

    # Choose default destination
    if destination_key is None:
        dest_dir = Path.cwd() / "outputs" if make_outputs_dir else Path.cwd()
        destination_key = dest_dir / src.name

    dst = Path(destination_key).resolve()

    # Avoid SameFileError
    if src == dst:
        print(f"[upload_result] Source and destination are the same ({src}); nothing to do.")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[upload_result] Copied {src} → {dst}")


# Optional main block for CLI use
if __name__ == "__main__":
    upload_to_s3("runs.zip")
