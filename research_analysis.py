"""Offline, family-aware analysis of reviewed FEVER-Symmetric responses.

Input is the immutable-response audit exported by ``rescore_results.py``.  No
model calls are made here.  Rates in the CSVs are proportions, not percentages.
An unresolved answer is never silently counted as a successful defence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random


BASE = Path(__file__).resolve().parent
LABELS = {"SUPPORTS", "REFUTES"}
STRENGTHS = ("weak", "medium", "strong")
MODEL_ORDER = ("GPT-OSS-20B", "Llama-3.1-8B", "Haiku-5.5", "Sonnet-5.5")


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def usable_label(row):
    """Pending review and noncanonical decisions remain unresolved."""
    if not row or str(row.get("needs_review", "")).lower() in {"true", "1", "yes"}:
        return None
    answer = row.get("label")
    return answer if answer in LABELS else None


def is_correct(row):
    return bool(row and usable_label(row) == row["ground_truth_label"])


def originally_attacked(row):
    value = row.get("original_attacked")
    if value is not None and value != "":
        return str(value).lower() in {"true", "1", "yes"}
    return row.get("original_label") == row["ground_truth_label"]


def is_original(row):
    return row is not None and row.get("source_kind", "original") != "supplemental"


def direction(row):
    return "R_to_S" if row["ground_truth_label"] == "REFUTES" else "S_to_R"


def response_status(row):
    if row is None:
        return "missing"
    if usable_label(row) is None:
        return "unresolved"
    return "retained" if is_correct(row) else "flipped"


def quantile(values, probability):
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    left = int(location)
    weight = location - left
    return ordered[left] * (1 - weight) + ordered[min(left + 1, len(ordered) - 1)] * weight


def cluster_bootstrap(clusters, statistic, resamples=2000, seed=2026):
    """Percentile CI by resampling whole families, recomputing denominators.

    ``clusters`` maps family IDs to additive vectors, with all sample variants
    and attack strengths already accumulated inside each vector.  Resampling
    these vectors keeps their dependence intact.  Undefined replicates are
    omitted and their count is returned.  Fewer than two clusters has no CI.
    """
    vectors = list(clusters.values())
    if len(vectors) < 2 or resamples < 1:
        return {"ci_low": None, "ci_high": None, "bootstrap_valid": 0,
                "bootstrap_clusters": len(vectors)}
    try:
        import numpy as np
    except ImportError:
        generator = random.Random(seed)
        totals = ([sum(column) for column in zip(*(generator.choice(vectors)
                   for _ in vectors))] for _ in range(resamples))
    else:
        generator = np.random.default_rng(seed)
        values = np.asarray(vectors, dtype=float)
        # Bounded memory for datasets larger than FEVER-Symmetric.
        def batches():
            for offset in range(0, resamples, 250):
                indices = generator.integers(0, len(vectors),
                                            size=(min(250, resamples - offset), len(vectors)))
                yield from values[indices].sum(axis=1)
        totals = batches()
    estimates = []
    for total in totals:
        value = statistic(total)
        if value is not None and math.isfinite(value):
            estimates.append(float(value))
    return {"ci_low": quantile(estimates, .025) if estimates else None,
            "ci_high": quantile(estimates, .975) if estimates else None,
            "bootstrap_valid": len(estimates), "bootstrap_clusters": len(vectors)}


def load_audit(path):
    required = {"model", "sample_id", "stage", "group_id", "variant", "display_id",
                "claim", "evidence_text", "ground_truth_label", "label", "format",
                "needs_review", "original_label"}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Audit columns missing: {sorted(missing)}")
        rows = list(reader)
    indexed = {}
    for row in rows:
        if row["stage"] not in ("baseline", *STRENGTHS):
            raise ValueError(f"Unexpected stage: {row['stage']}")
        if row["ground_truth_label"] not in LABELS:
            raise ValueError(f"Unexpected gold label: {row['ground_truth_label']}")
        key = (row["model"], row["sample_id"], row["stage"])
        if key in indexed:
            raise ValueError(f"Duplicate response in audit: {key}")
        indexed[key] = row
    for row in rows:
        if row["stage"] != "baseline":
            baseline = indexed.get((row["model"], row["sample_id"], "baseline"))
            if not baseline:
                raise ValueError(f"Attack without baseline: {row['model']} {row['sample_id']}")
            for key in ("group_id", "variant", "ground_truth_label", "claim", "evidence_text"):
                if row[key] != baseline[key]:
                    raise ValueError(f"Attack/baseline metadata mismatch in {key}: {row['sample_id']}")
    return rows


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _add_vector(clusters, group_id, values):
    if group_id not in clusters:
        clusters[group_id] = [0] * len(values)
    clusters[group_id] = [a + b for a, b in zip(clusters[group_id], values)]


def _ci(clusters, statistic, resamples, seed, prefix=""):
    return {prefix + key: value for key, value in
            cluster_bootstrap(clusters, statistic, resamples, seed).items()}


def rate_statistics(rows, expected=None):
    counts = Counter(response_status(row) for row in rows)
    observed = sum(counts[key] for key in ("flipped", "retained", "unresolved"))
    expected = len(rows) if expected is None else expected
    return {"expected_attacks": expected, "observed_attacks": observed,
            "missing_attacks": expected - observed,
            "flips": counts["flipped"], "retained": counts["retained"],
            "unresolved": counts["unresolved"],
            "confirmed_asr": ratio(counts["flipped"], observed),
            "asr_resolved_only": ratio(counts["flipped"], counts["flipped"] + counts["retained"]),
            "asr_upper_if_unresolved_flip": ratio(counts["flipped"] + counts["unresolved"], observed),
            "label_coverage": ratio(counts["flipped"] + counts["retained"], observed),
            "attack_coverage": ratio(observed, expected)}


def baseline_tables(model, baselines, resamples, seed):
    correct = sum(map(is_correct, baselines))
    resolved = sum(usable_label(row) is not None for row in baselines)
    summary = {"model": model, "baseline_samples": len(baselines), "correct": correct,
               "incorrect": resolved - correct, "unresolved": len(baselines) - resolved,
               "accuracy": ratio(correct, len(baselines)),
               "accuracy_resolved_only": ratio(correct, resolved),
               "label_coverage": ratio(resolved, len(baselines)),
               "original_parser_correct": sum(row["original_label"] == row["ground_truth_label"] for row in baselines),
               "baseline_labels_changed": sum((usable_label(row) or row["label"]) != row["original_label"] for row in baselines)}
    family_groups = defaultdict(dict)
    evidence_groups = defaultdict(list)
    accuracy_clusters, evidence_clusters = {}, {}
    for row in baselines:
        variant = row["variant"]
        if variant not in "abcd" or len(variant) != 1:
            raise ValueError(f"Unsupported symmetric variant: {variant}")
        if variant in family_groups[row["group_id"]]:
            raise ValueError(f"Duplicate family variant: {model} {row['display_id']}")
        family_groups[row["group_id"]][variant] = row
        evidence_type = "original" if variant in "ac" else "edited"
        evidence_groups[evidence_type].append(row)
        _add_vector(accuracy_clusters, row["group_id"], [int(is_correct(row)), 1])
        _add_vector(evidence_clusters, row["group_id"],
                    [int(is_correct(row)) if evidence_type == "original" else 0,
                     int(evidence_type == "original"),
                     int(is_correct(row)) if evidence_type == "edited" else 0,
                     int(evidence_type == "edited")])
    summary.update(_ci(accuracy_clusters, lambda v: ratio(v[0], v[1]), resamples, seed, "accuracy_"))
    family_rows = []
    complete_families = 0
    all_four = 0
    for group_id, members in sorted(family_groups.items()):
        complete = set(members) == set("abcd")
        if complete:
            if members["a"]["evidence_text"] != members["c"]["evidence_text"] or members["b"]["evidence_text"] != members["d"]["evidence_text"]:
                raise ValueError(f"Unexpected symmetric evidence structure: {model} {group_id}")
        all_correct = complete and all(is_correct(row) for row in members.values())
        complete_families += complete
        all_four += all_correct
        entry = {"model": model, "group_id": group_id, "observed_variants": len(members),
                 "complete_family": complete, "correct_variants": sum(map(is_correct, members.values())),
                 "unresolved_variants": sum(usable_label(row) is None for row in members.values()),
                 "all_four_correct": all_correct if complete else None}
        for variant in "abcd":
            row = members.get(variant)
            entry.update({f"{variant}_id": row["sample_id"] if row else "",
                          f"{variant}_gold": row["ground_truth_label"] if row else "",
                          f"{variant}_label": row["label"] if row else "MISSING",
                          f"{variant}_correct": is_correct(row) if row else None})
        family_rows.append(entry)
    summary.update({"observed_families": len(family_groups), "complete_families": complete_families,
                    "families_all_four_correct": all_four,
                    "all_four_correct_rate": ratio(all_four, complete_families)})
    family_clusters = {row["group_id"]: [int(row["all_four_correct"]), 1]
                       for row in family_rows if row["complete_family"]}
    summary.update(_ci(family_clusters, lambda v: ratio(v[0], v[1]), resamples, seed, "all_four_"))
    evidence_rows = []
    for evidence_type in ("original", "edited"):
        selected = evidence_groups[evidence_type]
        clusters = {}
        for row in selected:
            _add_vector(clusters, row["group_id"], [int(is_correct(row)), 1])
        entry = {"model": model, "evidence_type": evidence_type, "samples": len(selected),
                 "correct": sum(map(is_correct, selected)),
                 "unresolved": sum(usable_label(row) is None for row in selected),
                 "supports_gold": sum(row["ground_truth_label"] == "SUPPORTS" for row in selected),
                 "refutes_gold": sum(row["ground_truth_label"] == "REFUTES" for row in selected),
                 "accuracy": ratio(sum(map(is_correct, selected)), len(selected))}
        entry.update(_ci(clusters, lambda v: ratio(v[0], v[1]), resamples, seed))
        evidence_rows.append(entry)
    gap = {"model": model,
           "original_accuracy": evidence_rows[0]["accuracy"], "edited_accuracy": evidence_rows[1]["accuracy"],
           "gap_original_minus_edited": (evidence_rows[0]["accuracy"] - evidence_rows[1]["accuracy"]
                                          if all(row["accuracy"] is not None for row in evidence_rows) else None)}
    gap_stat = lambda v: v[0] / v[1] - v[2] / v[3] if v[1] and v[3] else None
    gap.update(_ci(evidence_clusters, gap_stat, resamples, seed))
    return summary, evidence_rows, gap, family_rows


def attack_tables(model, baselines, index, resamples, seed, cohort="original"):
    eligible = [row for row in baselines if is_correct(row) and
                (cohort != "original" or originally_attacked(row))]
    family_ids = {row["group_id"] for row in baselines}
    rates, gaps = [], []
    for strength in (*STRENGTHS, "pooled"):
        stages = STRENGTHS if strength == "pooled" else (strength,)
        directional_clusters = {group_id: [0, 0, 0, 0] for group_id in sorted(family_ids)}
        for direction_name in ("ALL", "R_to_S", "S_to_R"):
            subset = [row for row in eligible if direction_name == "ALL" or direction(row) == direction_name]
            attacks, clusters = [], {group_id: [0, 0] for group_id in sorted(family_ids)}
            for baseline in subset:
                for stage in stages:
                    attack = index.get((model, baseline["sample_id"], stage))
                    if cohort == "original" and not is_original(attack):
                        attack = None
                    attacks.append(attack)
                    if attack is not None:
                        flipped = response_status(attack) == "flipped"
                        _add_vector(clusters, baseline["group_id"], [int(flipped), 1])
                        if direction_name == "ALL":
                            values = [int(flipped), 1, 0, 0] if direction(baseline) == "R_to_S" else [0, 0, int(flipped), 1]
                            _add_vector(directional_clusters, baseline["group_id"], values)
            entry = {"model": model, "cohort": cohort, "strength": strength, "direction": direction_name,
                     "baseline_correct_samples": len(subset), **rate_statistics(attacks)}
            entry["supplemental_attacks"] = sum(row is not None and not is_original(row) for row in attacks)
            entry["newly_correct_baseline_samples"] = sum(is_correct(row) and not originally_attacked(row)
                                                           for row in baselines
                                                           if direction_name == "ALL" or direction(row) == direction_name)
            entry["originally_attacked_reclassified_baseline_samples"] = sum(originally_attacked(row) and not is_correct(row)
                                                                             for row in baselines
                                                                             if direction_name == "ALL" or direction(row) == direction_name)
            entry.update(_ci(clusters, lambda v: ratio(v[0], v[1]), resamples, seed))
            rates.append(entry)
        r_to_s, s_to_r = rates[-2], rates[-1]
        gap = {"model": model, "cohort": cohort, "strength": strength,
               "r_to_s": r_to_s["confirmed_asr"], "s_to_r": s_to_r["confirmed_asr"],
               "gap_r_to_s_minus_s_to_r": (r_to_s["confirmed_asr"] - s_to_r["confirmed_asr"]
                                             if r_to_s["confirmed_asr"] is not None and s_to_r["confirmed_asr"] is not None else None),
               "r_unresolved": r_to_s["unresolved"], "s_unresolved": s_to_r["unresolved"]}
        gap.update(_ci(directional_clusters,
                       lambda v: v[0] / v[1] - v[2] / v[3] if v[1] and v[3] else None,
                       resamples, seed))
        gaps.append(gap)
    return rates, gaps


def matched_pair_rows(model, baselines, index, cohort="original"):
    """Match claims within a family; no unavailable attack is reconstructed."""
    families = defaultdict(lambda: defaultdict(list))
    for row in baselines:
        families[row["group_id"]][row["claim"]].append(row)
    rows = []
    for group_id, claims in sorted(families.items()):
        for claim, members in claims.items():
            if len(members) != 2:
                # A partially collected family must not create an invented pair.
                continue
            members.sort(key=lambda row: row["variant"])
            if {row["ground_truth_label"] for row in members} != LABELS:
                raise ValueError(f"Same-claim pair lacks opposite gold labels: {model} {group_id}")
            if members[0]["evidence_text"] == members[1]["evidence_text"]:
                raise ValueError(f"Same-claim pair lacks different evidence: {model} {group_id}")
            baseline_correct_both = all(map(is_correct, members))
            original_attacked_both = all(map(originally_attacked, members))
            eligible = baseline_correct_both and (cohort != "original" or original_attacked_both)
            for strength in STRENGTHS:
                attacks = [index.get((model, row["sample_id"], strength)) for row in members]
                if cohort == "original":
                    attacks = [row if is_original(row) else None for row in attacks]
                statuses = [response_status(row) for row in attacks]
                observed = all(row is not None for row in attacks)
                if not eligible:
                    outcome = "baseline_ineligible"
                elif not observed:
                    outcome = "missing"
                elif "unresolved" in statuses:
                    outcome = "unresolved"
                else:
                    outcome = {0: "both_correct", 1: "one_wrong", 2: "both_wrong"}[statuses.count("flipped")]
                row = {"model": model, "cohort": cohort, "group_id": group_id,
                       "pair_id": group_id + "".join(item["variant"] for item in members),
                       "claim": claim, "strength": strength, "baseline_eligible": eligible,
                       "baseline_correct_both": baseline_correct_both,
                       "original_attacked_both": original_attacked_both,
                       "supplemental_attacks": sum(attack is not None and not is_original(attack) for attack in attacks),
                       "both_attacks_observed": observed, "outcome": outcome,
                       "both_predictions_opposite": (usable_label(attacks[0]) != usable_label(attacks[1])
                                                      if all(usable_label(item) for item in attacks) else None)}
                for baseline, attack, status in zip(members, attacks, statuses):
                    prefix = direction(baseline).lower()
                    row.update({f"{prefix}_sample_id": baseline["sample_id"],
                                f"{prefix}_display_id": baseline["display_id"],
                                f"{prefix}_baseline_label": baseline["label"],
                                f"{prefix}_attack_label": attack["label"] if attack else "MISSING",
                                f"{prefix}_status": status,
                                f"{prefix}_evidence": baseline["evidence_text"]})
                rows.append(row)
    return rows


def matched_tables(model, pair_rows, resamples, seed, cohort="original"):
    summaries, directional = [], []
    for strength in (*STRENGTHS, "pooled"):
        selected = [row for row in pair_rows if strength == "pooled" or row["strength"] == strength]
        eligible = [row for row in selected if row["baseline_eligible"]]
        counts = Counter(row["outcome"] for row in eligible)
        observed = [row for row in eligible if row["both_attacks_observed"]]
        fully_resolved = [row for row in observed if row["outcome"] != "unresolved"]
        summary = {"model": model, "cohort": cohort, "strength": strength, "candidate_pairs": len(selected),
                   "baseline_eligible_pairs": len(eligible), "baseline_ineligible_pairs": len(selected) - len(eligible),
                   "corrected_baseline_both_correct_pairs": sum(row["baseline_correct_both"] for row in selected),
                   "both_attacks_observed": len(observed), "fully_resolved_pairs": len(fully_resolved),
                   "supplemental_attack_responses": sum(row["supplemental_attacks"] for row in eligible)}
        for outcome in ("both_correct", "one_wrong", "both_wrong", "unresolved", "missing"):
            summary[outcome] = counts[outcome]
            summary[outcome + "_rate"] = ratio(counts[outcome], len(eligible))
        summary["both_wrong_but_predictions_opposite"] = sum(row["outcome"] == "both_wrong" and row["both_predictions_opposite"] for row in eligible)
        summaries.append(summary)
        # Both directions use exactly the same observed pairs.  An unresolved
        # output remains in the denominator and is reported separately.
        family_ids = sorted({row["group_id"] for row in selected})
        clusters = {group_id: [0, 0, 0, 0] for group_id in family_ids}
        r_clusters = {group_id: [0, 0] for group_id in family_ids}
        s_clusters = {group_id: [0, 0] for group_id in family_ids}
        direction_counts = {}
        for name in ("r_to_s", "s_to_r"):
            direction_counts[name] = Counter(row[name + "_status"] for row in observed)
        for row in observed:
            r_flip, s_flip = int(row["r_to_s_status"] == "flipped"), int(row["s_to_r_status"] == "flipped")
            _add_vector(clusters, row["group_id"], [r_flip, 1, s_flip, 1])
            _add_vector(r_clusters, row["group_id"], [r_flip, 1])
            _add_vector(s_clusters, row["group_id"], [s_flip, 1])
        entry = {"model": model, "cohort": cohort, "strength": strength, "matched_observed_pairs": len(observed),
                 "eligible_pairs_missing_attacks": counts["missing"], "unresolved_pairs": counts["unresolved"]}
        for name in ("r_to_s", "s_to_r"):
            counts_direction = direction_counts[name]
            flips = counts_direction["flipped"]
            entry.update({name: ratio(flips, len(observed)), name + "_flips": flips,
                          name + "_unresolved": counts_direction["unresolved"],
                          name + "_asr_resolved_only": ratio(flips, flips + counts_direction["retained"]),
                          name + "_upper_if_unresolved_flip": ratio(flips + counts_direction["unresolved"], len(observed))})
        entry["gap_r_to_s_minus_s_to_r"] = (entry["r_to_s"] - entry["s_to_r"] if observed else None)
        entry.update(_ci(clusters, lambda v: v[0] / v[1] - v[2] / v[3] if v[1] and v[3] else None,
                         resamples, seed))
        entry.update(_ci(r_clusters, lambda v: ratio(v[0], v[1]), resamples, seed, "r_to_s_"))
        entry.update(_ci(s_clusters, lambda v: ratio(v[0], v[1]), resamples, seed, "s_to_r_"))
        directional.append(entry)
    return summaries, directional


def common_correct_tables(models, baseline_by_model, index, resamples=2000, seed=2026,
                          comparison_group="all_models"):
    available = [{row["sample_id"] for row in baseline_by_model[model]} for model in models]
    correct_sets = [{row["sample_id"] for row in baseline_by_model[model] if is_correct(row)} for model in models]
    queried_correct_sets = [{row["sample_id"] for row in baseline_by_model[model]
                             if is_correct(row) and originally_attacked(row)} for model in models]
    common_available = set.intersection(*available) if available else set()
    all_common_correct = set.intersection(*correct_sets) if correct_sets else set()
    common_correct = set.intersection(*queried_correct_sets) if queried_correct_sets else set()
    reference = {row["sample_id"]: row for row in baseline_by_model[models[0]]} if models else {}
    sample_rows = [{"comparison_group": comparison_group, "sample_id": sample_id, "display_id": reference[sample_id]["display_id"],
                    "group_id": reference[sample_id]["group_id"],
                    "ground_truth_label": reference[sample_id]["ground_truth_label"],
                    "claim": reference[sample_id]["claim"],
                    "all_models_baseline_correct": sample_id in all_common_correct,
                    "all_models_primary_eligible": sample_id in common_correct}
                   for sample_id in sorted(common_available, key=lambda item: reference[item]["display_id"])]
    summaries = []
    for model in models:
        for strength in (*STRENGTHS, "pooled"):
            stages = STRENGTHS if strength == "pooled" else (strength,)
            attacks = [index.get((model, sample_id, stage)) for sample_id in sorted(common_correct) for stage in stages]
            attacks = [row if is_original(row) else None for row in attacks]
            entry = {"model": model, "comparison_group": comparison_group, "strength": strength,
                     "models_in_intersection": " | ".join(models),
                     "common_baseline_available": len(common_available),
                     "common_baseline_correct": len(common_correct),
                     "common_correct_including_newly_eligible": len(all_common_correct), **rate_statistics(attacks)}
            directional_clusters = {group_id: [0, 0, 0, 0]
                                    for group_id in sorted({reference[sample_id]["group_id"] for sample_id in common_available})}
            for name in ("R_to_S", "S_to_R"):
                sample_ids = [sample_id for sample_id in sorted(common_correct) if direction(reference[sample_id]) == name]
                directional_attacks = [index.get((model, sample_id, stage)) for sample_id in sample_ids for stage in stages]
                directional_attacks = [row if is_original(row) else None for row in directional_attacks]
                statistics = rate_statistics(directional_attacks)
                prefix = name.lower()
                entry[prefix] = statistics["confirmed_asr"]
                for key in ("flips", "retained", "unresolved", "observed_attacks", "missing_attacks"):
                    entry[prefix + "_" + key] = statistics[key]
                for attack in directional_attacks:
                    if attack is None:
                        continue
                    flip = int(response_status(attack) == "flipped")
                    _add_vector(directional_clusters, attack["group_id"],
                                [flip, 1, 0, 0] if name == "R_to_S" else [0, 0, flip, 1])
            entry["gap_r_to_s_minus_s_to_r"] = (entry["r_to_s"] - entry["s_to_r"]
                                                if entry["r_to_s"] is not None and entry["s_to_r"] is not None else None)
            entry.update(_ci(directional_clusters, lambda v: v[0] / v[1] - v[2] / v[3] if v[1] and v[3] else None,
                             resamples, seed))
            summaries.append(entry)
    return sample_rows, summaries


def scoring_comparison_tables(rows, models, baseline_by_model):
    """Parser sensitivity on identical original response records, no resampling."""
    output = []
    baseline_index = {(row["model"], row["sample_id"]): row
                      for values in baseline_by_model.values() for row in values}
    for model in models:
        for stage in ("baseline", *STRENGTHS, "pooled_attacks"):
            selected = [row for row in rows if row["model"] == model and is_original(row) and
                        (row["stage"] != "baseline" if stage == "pooled_attacks" else row["stage"] == stage)]
            original_wrong = sum(row["original_label"] in LABELS and row["original_label"] != row["ground_truth_label"] for row in selected)
            revised_wrong = sum(usable_label(row) is not None and not is_correct(row) for row in selected)
            eligible = [row for row in selected if stage == "baseline" or
                        is_correct(baseline_index[(model, row["sample_id"])])]
            output.append({"model": model, "stage": stage, "same_original_responses": len(selected),
                           "original_correct": sum(row["original_label"] == row["ground_truth_label"] for row in selected),
                           "original_wrong": original_wrong,
                           "original_unresolved": sum(row["original_label"] not in LABELS for row in selected),
                           "revised_correct": sum(map(is_correct, selected)), "revised_wrong": revised_wrong,
                           "revised_unresolved": sum(usable_label(row) is None for row in selected),
                           "original_wrong_rate_same_responses": ratio(original_wrong, len(selected)),
                           "revised_wrong_rate_same_responses": ratio(revised_wrong, len(selected)),
                           "changed_labels": sum((usable_label(row) or row["label"]) != row["original_label"] for row in selected),
                           "responses_with_corrected_baseline": len(eligible),
                           "revised_wrong_with_corrected_baseline": sum(usable_label(row) is not None and not is_correct(row) for row in eligible)})
    return output


def analyze(rows, resamples=2000, seed=2026):
    """Compute the paper tables and the supplemental coverage check."""
    models = sorted({row["model"] for row in rows}, key=lambda model: (MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER), model))
    index = {(row["model"], row["sample_id"], row["stage"]): row for row in rows}
    baselines = {model: [row for row in rows if row["model"] == model and row["stage"] == "baseline"] for model in models}
    tables = {key: [] for key in ("baseline_summary", "baseline_by_evidence", "evidence_gap_summary",
                                  "aggregate_attacks", "directional_gaps", "matched_pairs", "matched_pair_summary",
                                  "matched_directional_summary", "corrected_coverage_attacks")}
    for model in models:
        summary, evidence_rows, gap, _ = baseline_tables(model, baselines[model], resamples, seed)
        tables["baseline_summary"].append(summary)
        tables["baseline_by_evidence"].extend(evidence_rows)
        tables["evidence_gap_summary"].append(gap)
        attacks, gaps = attack_tables(model, baselines[model], index, resamples, seed)
        tables["aggregate_attacks"].extend(attacks)
        tables["directional_gaps"].extend(gaps)
        pairs = matched_pair_rows(model, baselines[model], index)
        pair_summary, pair_direction = matched_tables(model, pairs, resamples, seed)
        tables["matched_pairs"].extend(pairs)
        tables["matched_pair_summary"].extend(pair_summary)
        tables["matched_directional_summary"].extend(pair_direction)
        corrected_attacks, _ = attack_tables(model, baselines[model], index, resamples, seed, "corrected_coverage")
        tables["corrected_coverage_attacks"].extend(corrected_attacks)
    tables["common_correct_samples"], tables["common_correct_summary"] = common_correct_tables(models, baselines, index, resamples, seed)
    local_models = [model for model in models if model in MODEL_ORDER[:2]]
    if len(local_models) == 2 and len(models) > 2:
        samples, summary = common_correct_tables(local_models, baselines, index, resamples, seed, "original_local_models")
        tables["common_correct_samples"].extend(samples)
        tables["common_correct_summary"].extend(summary)
    tables["scoring_comparison"] = scoring_comparison_tables(rows, models, baselines)
    return tables


def run(input_path, output_dir, resamples=2000, seed=2026):
    rows = load_audit(input_path)
    if not rows:
        raise ValueError("The audit has no responses.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = analyze(rows, resamples, seed)
    for name, content in tables.items():
        write_csv(output_dir / f"{name}.csv", content)
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "audit_file": str(Path(input_path).resolve()),
        "audit_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        "responses": len(rows), "bootstrap_resamples": resamples, "bootstrap_seed": seed,
        "definitions": {
            "rates": "CSV rates are proportions on [0,1]; gaps are signed proportion differences.",
            "usable_label": "SUPPORTS/REFUTES and needs_review=false; all others unresolved.",
            "accuracy": "Correct labels divided by all observed baselines; unresolved labels remain explicit.",
            "confirmed_asr": "Verified flips / observed eligible attacks; unresolved answers are separate, not retained answers.",
            "asr_resolved_only": "Verified flips / verified flips plus retained answers, excluding unresolved answers.",
            "missing_attacks": "Expected under revised baseline eligibility but never observed; never imputed.",
            "aggregate_eligibility": "Primary original cohort: revised baseline correct AND originally attacked. Only original attack responses used. Corrected-coverage tables also include newly eligible baselines and clearly marked supplemental responses.",
            "direction": "R_to_S is gold REFUTES attacked toward SUPPORTS; S_to_R the reverse.",
            "directional_gap": "R_to_S minus S_to_R, signed, for both aggregate and matched rates.",
            "original_evidence": "Variants a/c contain original evidence, b/d edited evidence, verified within complete families.",
            "evidence_gap": "Original evidence accuracy minus edited evidence accuracy.",
            "all_four_correct": "All four baseline variants correct divided by complete observed families.",
            "matched_pairs": "Same exact claim within a family, opposite gold labels, different evidence; both revised baselines correct and originally attacked for primary tables. Corrected-coverage tables retain any missing attacks and identify supplemental responses.",
            "matched_directional": "Exactly the same eligible pairs in both directions, requiring both attacks observed.",
            "bootstrap": "95% percentile interval; resample whole family clusters, recompute all denominators. Variants and strengths remain together.",
            "common_correct": "Primary intersection requires revised baseline correctness AND original attack eligibility for every compared model. Newly eligible baselines are counted separately, so original query coverage is identical across compared models.",
            "automatic_evaluator": "Automatic judge decisions are not represented as human validation. Inspect response audit methods and reasons.",
        },
        "limitations": [
            "Corrected labels cannot reconstruct attacks skipped during original collection.",
            "Original/edited evidence differences may reflect wording or annotation artifacts as well as conflict with model knowledge.",
            "Three attack templates vary several wording features; comparisons do not isolate a causal effect of pressure intensity.",
            "Common-cohort comparisons require revised correctness and original query eligibility in every compared model; they target that selected subset, not the full dataset.",
            "Bootstrap implementation uses NumPy when installed, otherwise Python random; reproducible within a fixed backend and seed.",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Analyzed {len(rows):,} saved responses from {len(tables['baseline_summary'])} models -> {output_dir}")
    return tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BASE / "results/review/response_audit.csv")
    parser.add_argument("--output-dir", type=Path, default=BASE / "results/analysis")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    if args.resamples < 0:
        parser.error("--resamples must be nonnegative")
    run(args.input, args.output_dir, args.resamples, args.seed)


if __name__ == "__main__":
    main()
