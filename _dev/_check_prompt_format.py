"""Sanity check for weight_generator_v1.txt — placeholder format only."""
import json
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "dsc_framework" / "prompts" / "weight_generator_v1.txt"
template = p.read_text(encoding="utf-8")

rendered = template.format(
    data_type="tabular",
    task="classification",
    metric_names_json=json.dumps(["completeness", "uniqueness", "class_balance"]),
    dataset_metadata_json=json.dumps({"n_rows": 1000, "n_cols": 8}, indent=2),
    w_min=0.01,
    w_max=0.60,
)
print("prompt format OK")
print("rendered len:", len(rendered))
print("---")
print(rendered[:600])
