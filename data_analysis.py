# 
# Calculate baseline accuracy, ASR by attack strength and direction,
# and directional asymmetry (R_to_S - S_to_R).
# ASR = successful flips / all attempted attacks (including INVALID responses).
# Save summary tables and figures to analysis_results/.

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "analysis_results"
OUT.mkdir(exist_ok=True)

FILES = {
    "GPT-OSS-20B": "pilot_results_gpt-oss-20b.csv",
    "Llama-3.1-8B": "pilot_results_llama-3.1-8b.csv",
}
STRENGTHS = ["weak", "medium", "strong"]

baseline_rows = []
attack_rows = []

for model, filename in FILES.items():
    path = BASE / filename
    if not path.exists():
        path = BASE / "results" / filename
    df = pd.read_csv(path, dtype={"sample_id": str})

    baseline = df.drop_duplicates("sample_id")
    correct = baseline["initial_answer"] == baseline["ground_truth_label"]
    baseline_rows.append({
        "model": model,
        "samples": len(baseline),
        "correct": int(correct.sum()),
        "accuracy_pct": round(correct.mean() * 100, 2),
    })

    attacks = df[df["strength"].isin(STRENGTHS)].copy()
    attacks["direction"] = attacks["ground_truth_label"].map({
        "SUPPORTS": "S_to_R",
        "REFUTES": "R_to_S",
    })
    attacks["success"] = attacks["attack_result"].eq("SUCCESS")
    attacks["invalid"] = attacks["attack_result"].str.startswith("INVALID")

    grouped = attacks.groupby(["strength", "direction"], observed=True)
    for (strength, direction), group in grouped:
        attack_rows.append({
            "model": model,
            "strength": strength,
            "direction": direction,
            "attempts": len(group),
            "success": int(group["success"].sum()),
            "invalid": int(group["invalid"].sum()),
            "asr_pct": round(group["success"].mean() * 100, 2),
        })

baseline_table = pd.DataFrame(baseline_rows)
asr_table = pd.DataFrame(attack_rows)
asr_table["strength"] = pd.Categorical(
    asr_table["strength"], categories=STRENGTHS, ordered=True
)
asr_table = asr_table.sort_values(["model", "strength", "direction"])

gap_table = asr_table.pivot(
    index=["model", "strength"], columns="direction", values="asr_pct"
).reset_index()
gap_table["difference_pp"] = gap_table["R_to_S"] - gap_table["S_to_R"]

baseline_table.to_csv(OUT / "baseline_summary.csv", index=False)
asr_table.to_csv(OUT / "asr_by_direction_strength.csv", index=False)
gap_table.to_csv(OUT / "directional_asymmetry.csv", index=False)

for model in FILES:
    part = asr_table[asr_table["model"] == model]
    plt.figure(figsize=(6, 4))
    for direction in ["S_to_R", "R_to_S"]:
        values = part[part["direction"] == direction].set_index("strength")
        values = values.reindex(STRENGTHS)
        plt.plot(STRENGTHS, values["asr_pct"], marker="o", label=direction)
    plt.title(model)
    plt.xlabel("Attack strength")
    plt.ylabel("ASR (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / f"asr_{model.lower().replace('-', '_').replace('.', '_')}.png", dpi=200)
    plt.close()

print("\nBaseline accuracy:\n", baseline_table.to_string(index=False))
print("\nASR by direction and strength:\n", asr_table.to_string(index=False))
print("\nDirectional asymmetry (R_to_S - S_to_R):\n", gap_table.to_string(index=False))
print(f"\nSaved to: {OUT}")
