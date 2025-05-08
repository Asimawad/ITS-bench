import os
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
# This makes imports work regardless of how the script is called
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now import will work
from environment.utils_zip import make_filtered_zip

if __name__ == "__main__":
    # --- Step 1: Set up paths ---
    local_folder_to_zip = "runs"
    output_directory = "RESULTS_REPORTS"

    # --- Step 2: Ensure the output directory exists ---
    if not os.path.exists(output_directory):
        print(f"[INFO] Creating directory: {output_directory}")
        os.makedirs(output_directory)
    else:
        print(f"[INFO] Output directory already exists: {output_directory}")

    # --- Step 3: Define zip file name ---
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    zip_base_name = f"runs_{timestamp}"
    # Full path in output directory
    zip_output_path = os.path.join(output_directory, zip_base_name)

    # --- Step 4: Check if the source folder exists ---
    if not os.path.isdir(local_folder_to_zip):
        print(f"[ERROR] Source folder '{local_folder_to_zip}' not found. Nothing to zip.")
    else:
        # --- Step 5: Use make_filtered_zip to create the archive ---
        print(
            f"[INFO] Using make_filtered_zip to archive: {local_folder_to_zip} → {zip_output_path}.zip"
        )

        # Call make_filtered_zip to create the archive
        # It will handle excluding 'input' and 'data' directories according to EXCLUDE_GLOBS
        # and will handle checking for file existence
        result_path = make_filtered_zip(zip_output_path, local_folder_to_zip)

        print(f"[INFO] ZIP archive created at: {result_path}")
