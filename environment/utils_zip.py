# utils_zip.py
import fnmatch
import zipfile
from pathlib import Path

EXCLUDE_GLOBS = [
    # skip the public data folder shipped with each run
    "*/*/home/data/**",  #  …/<run_id>/home/data/…
    # skip the resolved copy under each workspace
    "*/*/home/workspaces/*/input/**",  #  …/workspaces/<ws_id>/input/…
]


def _should_exclude(rel_path: Path) -> bool:
    """Return True iff *rel_path* (POSIX string) matches one of the globs."""
    path_str = rel_path.as_posix()
    return any(fnmatch.fnmatch(path_str, pat) for pat in EXCLUDE_GLOBS)


def make_filtered_zip(zip_name: str | Path, root_dir: str | Path) -> Path:
    """
    Create *zip_name* containing everything under *root_dir* **except**
    the paths covered by EXCLUDE_GLOBS.  Returns the Path to the archive.
    """
    zip_name = Path(zip_name).with_suffix(".zip")
    root_dir = Path(root_dir).resolve()

    files_processed = 0
    files_skipped_excluded = 0
    files_skipped_not_found = 0

    with zipfile.ZipFile(zip_name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in root_dir.rglob("*"):
            # Skip directories; rglob will still visit their children
            if path.is_dir():
                continue

            rel = path.relative_to(root_dir)
            if _should_exclude(rel):
                # comment out the next line if you don't want log spam
                # print(f"   ↷  skipping {rel}")
                files_skipped_excluded += 1
                continue

            # Check if the file still exists before adding to zip
            if not path.exists():
                # print(f"[WARNING] File no longer exists, skipping: {rel}")
                files_skipped_not_found += 1
                continue

            try:
                zf.write(path, rel)
                files_processed += 1
            except FileNotFoundError:
                # Double-check in case file was deleted between check and write
                # print(f"[WARNING] File disappeared before writing, skipping: {rel}")
                files_skipped_not_found += 1
            except Exception as e:
                print(f"[WARNING] Error adding file {rel} to zip: {e}")
                files_skipped_not_found += 1

    print(f"[zip] Wrote {zip_name} (root: {root_dir})")
    print(
        f"[zip] Summary: {files_processed} files added, {files_skipped_excluded} excluded, {files_skipped_not_found} not found"
    )
    return zip_name
