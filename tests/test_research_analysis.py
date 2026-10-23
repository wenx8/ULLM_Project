import csv
from pathlib import Path
import tempfile
import unittest

import research_analysis as analysis


def family(group="001", model="Haiku-5.5", correct=True):
    rows = []
    for variant, gold in zip("abcd", ("SUPPORTS", "REFUTES", "REFUTES", "SUPPORTS")):
        label = gold if correct else ("SUPPORTS" if gold == "REFUTES" else "REFUTES")
        rows.append({"model": model, "sample_id": group + variant, "group_id": group,
                     "variant": variant, "display_id": group + variant, "stage": "baseline",
                     "claim": f"{group}: " + ("original claim" if variant in "ab" else "edited claim"),
                     "evidence_text": "original evidence" if variant in "ac" else "edited evidence",
                     "ground_truth_label": gold, "label": label, "original_label": label,
                     "format": "single_label", "needs_review": "false", "original_attacked": str(correct),
                     "source_kind": "original", "raw_response": label})
    return rows


def attack(baseline, stage="weak", flipped=False, **updates):
    label = baseline["ground_truth_label"]
    if flipped:
        label = "SUPPORTS" if label == "REFUTES" else "REFUTES"
    return {**baseline, "stage": stage, "label": label, "original_label": label,
            "raw_response": label, **updates}


def index(rows):
    return {(row["model"], row["sample_id"], row["stage"]): row for row in rows}


class ResearchAnalysisTests(unittest.TestCase):
    def test_common_cohort_excludes_newly_correct_but_unqueried_baseline(self):
        first = family(model='Haiku-5.5')
        second = family(model='Sonnet-5.5')
        second[0]['original_attacked'] = 'false'
        rows = first + second + [attack(row) for row in first + second if row['original_attacked'] == 'True']
        _, summaries = analysis.common_correct_tables(
            ['Haiku-5.5', 'Sonnet-5.5'], {'Haiku-5.5': first, 'Sonnet-5.5': second}, index(rows), 20, 1)
        weak = [row for row in summaries if row['strength'] == 'weak']
        self.assertTrue(all(row['common_baseline_correct'] == 3 for row in weak))
        self.assertTrue(all(row['common_correct_including_newly_eligible'] == 4 for row in weak))
        self.assertTrue(all(row['missing_attacks'] == 0 for row in weak))

    def test_both_flips_preserve_opposite_predictions_but_are_not_robust(self):
        baseline = family()
        rows = baseline + [attack(row, flipped=True) for row in baseline]
        pairs = analysis.matched_pair_rows("Haiku-5.5", baseline, index(rows))
        weak = [row for row in pairs if row["strength"] == "weak"]
        self.assertEqual([row["outcome"] for row in weak], ["both_wrong", "both_wrong"])
        self.assertTrue(all(row["both_predictions_opposite"] for row in weak))
        summary, directional = analysis.matched_tables("Haiku-5.5", pairs, 20, 1)
        self.assertEqual(summary[0]["both_wrong_but_predictions_opposite"], 2)
        self.assertEqual(directional[0]["r_to_s"], 1)
        self.assertEqual(directional[0]["s_to_r"], 1)

    def test_missing_attack_remains_missing_and_wrong_baseline_is_ineligible(self):
        baseline = family()
        baseline[2]["label"] = "SUPPORTS"  # Wrong c baseline excludes c/d pair.
        rows = baseline + [attack(baseline[0])]
        pairs = analysis.matched_pair_rows("Haiku-5.5", baseline, index(rows))
        weak = [row for row in pairs if row["strength"] == "weak"]
        self.assertEqual(weak[0]["outcome"], "missing")
        self.assertEqual(weak[1]["outcome"], "baseline_ineligible")
        summary, directional = analysis.matched_tables("Haiku-5.5", pairs, 20, 1)
        self.assertEqual(summary[0]["baseline_eligible_pairs"], 1)
        self.assertEqual(summary[0]["missing"], 1)
        self.assertEqual(summary[0]["both_correct"], 0)
        self.assertIsNone(directional[0]["gap_r_to_s_minus_s_to_r"])

    def test_ambiguous_and_pending_review_are_not_retained_answers(self):
        baseline = family()
        attacks = [attack(baseline[0], label="AMBIGUOUS", needs_review="true", format="ambiguous"),
                   attack(baseline[1], label="REFUTES", needs_review="true"),
                   attack(baseline[2]), attack(baseline[3])]
        statistics = analysis.rate_statistics(attacks)
        self.assertEqual(statistics["unresolved"], 2)
        self.assertEqual(statistics["retained"], 2)
        self.assertEqual(statistics["asr_upper_if_unresolved_flip"], .5)
        pairs = analysis.matched_pair_rows("Haiku-5.5", baseline, index(baseline + attacks))
        self.assertEqual(pairs[0]["outcome"], "unresolved")

    def test_cluster_bootstrap_never_splits_a_family(self):
        observed = []
        def statistic(total):
            observed.append(tuple(total))
            return total[0] / total[1]
        result = analysis.cluster_bootstrap({"a": [1, 10, 3], "b": [100, 1000, 8]}, statistic, 100, 9)
        self.assertEqual(result["bootstrap_clusters"], 2)
        self.assertEqual(result["bootstrap_valid"], 100)
        self.assertTrue(set(observed) <= {(2, 20, 6), (101, 1010, 11), (200, 2000, 16)})
        self.assertAlmostEqual(result["ci_low"], .1)
        self.assertEqual(result, analysis.cluster_bootstrap({"a": [1, 10, 3], "b": [100, 1000, 8]}, statistic, 100, 9))
        self.assertIsNone(analysis.cluster_bootstrap({"a": [1, 1]}, lambda v: v[0] / v[1])["ci_low"])

    def test_directional_gap_sign_is_r_to_s_minus_s_to_r(self):
        baseline = family("001") + family("002")
        for row in baseline:
            if row["variant"] in "cd":
                row["label"] = "SUPPORTS" if row["ground_truth_label"] == "REFUTES" else "REFUTES"
        attacks = [attack(row, flipped=row["variant"] == "b" or row["display_id"] == "001a")
                   for row in baseline if row["variant"] in "ab"]
        rows = baseline + attacks
        aggregate, gaps = analysis.attack_tables("Haiku-5.5", baseline, index(rows), 50, 1)
        self.assertEqual(gaps[0]["gap_r_to_s_minus_s_to_r"], .5)
        pairs = analysis.matched_pair_rows("Haiku-5.5", baseline, index(rows))
        _, matched = analysis.matched_tables("Haiku-5.5", pairs, 50, 1)
        self.assertEqual(matched[0]["gap_r_to_s_minus_s_to_r"], .5)
        self.assertEqual(matched[0]["r_to_s"], 1)
        self.assertEqual(matched[0]["s_to_r"], .5)

    def test_original_edited_gap_and_all_four_family_accuracy(self):
        baseline = family("001") + family("002")
        for row in baseline:
            if row["variant"] in "bd":
                row["label"] = "SUPPORTS" if row["ground_truth_label"] == "REFUTES" else "REFUTES"
        summary, evidence, gap, families = analysis.baseline_tables("Haiku-5.5", baseline, 50, 7)
        self.assertEqual(summary["accuracy"], .5)
        self.assertEqual(summary["all_four_correct_rate"], 0)
        self.assertEqual(gap["gap_original_minus_edited"], 1)
        self.assertEqual(gap["ci_low"], 1)
        self.assertEqual(evidence[0]["supports_gold"], evidence[0]["refutes_gold"])
        self.assertEqual(len(families), 2)

    def test_supplements_and_newly_eligible_baselines_are_separate_cohorts(self):
        baseline = family()
        baseline[0].update(original_label="REFUTES", original_attacked="false")
        attacks = [attack(row, flipped=True, source_kind="supplemental" if row["variant"] == "a" else "original")
                   for row in baseline]
        all_rows = baseline + attacks
        primary, _ = analysis.attack_tables("Haiku-5.5", baseline, index(all_rows), 0, 1)
        expanded, _ = analysis.attack_tables("Haiku-5.5", baseline, index(all_rows), 0, 1, "corrected_coverage")
        self.assertEqual(primary[0]["observed_attacks"], 3)
        self.assertEqual(primary[0]["supplemental_attacks"], 0)
        self.assertEqual(expanded[0]["observed_attacks"], 4)
        self.assertEqual(expanded[0]["supplemental_attacks"], 1)
        primary_pairs = analysis.matched_pair_rows("Haiku-5.5", baseline, index(all_rows))
        expanded_pairs = analysis.matched_pair_rows("Haiku-5.5", baseline, index(all_rows), "corrected_coverage")
        self.assertEqual(primary_pairs[0]["outcome"], "baseline_ineligible")
        self.assertEqual(expanded_pairs[0]["outcome"], "both_wrong")

    def test_common_correct_controls_baseline_subset_and_reports_direction(self):
        models = ("Haiku-5.5", "Sonnet-5.5")
        baselines = {model: family(model=model) for model in models}
        # Sonnet wrong on c: every common comparison must exclude that ID.
        baselines["Sonnet-5.5"][2]["label"] = "SUPPORTS"
        rows = [row for values in baselines.values() for row in values]
        rows += [attack(row, flipped=row["ground_truth_label"] == "REFUTES") for row in rows.copy()]
        samples, summary = analysis.common_correct_tables(models, baselines, index(rows), 0, 1)
        weak = [row for row in summary if row["strength"] == "weak"]
        self.assertEqual(len(samples), 4)
        self.assertTrue(all(row["common_baseline_correct"] == 3 for row in weak))
        self.assertTrue(all(row["gap_r_to_s_minus_s_to_r"] == 1 for row in weak))

    def test_offline_end_to_end_exports_auditable_tables(self):
        baseline = family("001") + family("002")
        rows = baseline + [attack(row, stage, flipped=row["variant"] == "a")
                           for row in baseline for stage in analysis.STRENGTHS]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            audit = directory / "audit.csv"
            analysis.write_csv(audit, rows)
            tables = analysis.run(audit, directory / "out", resamples=20)
            self.assertEqual(tables["baseline_summary"][0]["baseline_samples"], 8)
            self.assertEqual(len(tables["matched_pairs"]), 12)
            self.assertFalse((directory / "out/analysis_summary.json").exists())
            self.assertTrue((directory / "out/metadata.json").exists())
            with (directory / "out/matched_pairs.csv").open(encoding="utf-8-sig", newline="") as stream:
                exported = list(csv.DictReader(stream))
            self.assertEqual(len(exported), 12)


if __name__ == "__main__":
    unittest.main()
