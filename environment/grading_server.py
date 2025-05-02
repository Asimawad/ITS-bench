import argparse
import os
from pathlib import Path

from flask import Flask, jsonify, request

from mlebench.grade import validate_submission
from mlebench.registry import registry

app = Flask(__name__)

PRIVATE_DATA_DIR = "/private/data"
COMPETITION_ID = os.getenv("COMPETITION_ID")  # This is populated for us at container runtime


def run_validation(submission: Path) -> str:
    new_registry = registry.set_data_dir(Path(PRIVATE_DATA_DIR))
    competition = new_registry.get_competition(COMPETITION_ID)
    is_valid, message = validate_submission(submission, competition)
    return message


@app.route("/validate", methods=["POST"])
def validate():
    submission_file = request.files["file"]
    submission_path = Path("/tmp/submission_to_validate.csv")
    submission_file.save(submission_path)

    try:
        result = run_validation(submission_path)
    except Exception as e:
        # Server error
        return jsonify({"error": "An unexpected error occurred.", "details": str(e)}), 500

    return jsonify({"result": result})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"}), 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grading server.")
    parser.add_argument(
        "--port",
        type=int,
        default=5000,  # Default port if --port is not provided
        help="Port to run the server on",
    )

    args = parser.parse_args()

    app.run(host="0.0.0.0", port=args.port, debug=False)  # *** Ensure args.port is used here ***
