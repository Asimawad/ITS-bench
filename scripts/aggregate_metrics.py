import json
import os

import numpy as np

# List your grading report files here
DISPLAY_NAME = "RedHatAI_DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic_data_None_25_steps"
# "gpt-4o-mini_data_None_25_steps"
# Metrics to aggregate: (Display Name, JSON Key)
metrics = [
    ("Made Submission", "submission_exists"),
    ("Valid Submission", "valid_submission"),
    ("Above Median", "above_median"),
    ("Bronze", "bronze_medal"),
    ("Silver", "silver_medal"),
    ("Gold", "gold_medal"),
    ("Any Medal", "any_medal"),
]

# Collect results for each metric across seeds
results = {name: [] for name, _ in metrics}

for file in os.listdir(f"{DISPLAY_NAME}/evaluation_results"):
    if "_all_" in file:
        print("skipping", file)
        continue
    with open(f"{DISPLAY_NAME}/evaluation_results/{file}", "r") as f:
        print("loading", file)
        data = json.load(f)
    reports = data["competition_reports"]
    total = len(reports)
    for name, key in metrics:
        count = sum(1 for r in reports if r[key])
        percent = 100.0 * count / total
        results[name].append(percent)

# Calculate mean and standard error for each metric
output = ["# Results Summary", "", "| Metric | Mean ± StdErr |", "|--------|--------------|"]
for name, _ in metrics:
    vals = np.array(results[name])
    mean = np.mean(vals)
    stderr = np.std(vals, ddof=1) / np.sqrt(len(vals))
    output.append(f"| {name} | {mean:.1f} ± {stderr:.1f} |")
    print(f"| {name} | {mean:.1f} ± {stderr:.1f} |")
# Save results to markdown file
with open(f"{DISPLAY_NAME}/results_summary.md", "w") as f:
    f.write("\n".join(output))
