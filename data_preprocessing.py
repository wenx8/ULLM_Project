
import json
import random
from pathlib import Path

# 1. File paths

BASE = Path(__file__).parent
DATA = BASE / "data"

dev_path = DATA / "raw" / "fever_symmetric_dev.jsonl"
output_path = DATA / "processed" / "pilot_fever_symmetric.jsonl"

output_path.parent.mkdir(parents=True, exist_ok=True)


# 2. Load FEVER-Symmetric
with open(dev_path, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f if line.strip()]

supports = [x for x in dataset if x["label"] == "SUPPORTS"]
refutes = [x for x in dataset if x["label"] == "REFUTES"]

print("Total samples:", len(dataset))
print("SUPPORTS:", len(supports))
print("REFUTES:", len(refutes))


# 3. Select pilot samples

rng = random.Random(42)

samples = (
    rng.sample(supports, len(supports))
    + rng.sample(refutes, len(refutes))
)

rng.shuffle(samples)


# 4. Preprocess data

processed_data = []

for sample in samples:
    evidence = sample["evidence"]

    processed_data.append({
        "id": sample["id"],
        "claim": sample["claim"],
        "label": sample["label"],
        "evidence_text": evidence,
        "evidence_length_words": len(evidence.split())
    })


# 5. Save processed dataset

with open(output_path, "w", encoding="utf-8") as f:
    for item in processed_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("\nProcessed samples:", len(processed_data))
print("Saved to:", output_path)

print("\nExample:")
print(json.dumps(processed_data[0], indent=2, ensure_ascii=False))
