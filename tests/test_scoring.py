import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from scoring import clean_response, parse_label, parse_response
from rescore_results import (AUDIT_FIELDS, JUDGE_SYSTEM, ExtractionJudge, canonical,
                             export_corrected, judge_cache_key, load_overrides,
                             load_responses, merge_supplemental,
                             score_observations, validate_judgment, write_csv)


def completion(text, stop='end_turn'):
    return SimpleNamespace(content=[SimpleNamespace(type='text', text=text)],
                           model='claude-haiku-5-5', stop_reason=stop,
                           usage=SimpleNamespace(model_dump=lambda: {'input_tokens': 100, 'output_tokens': 50}))


class ExtractionTests(unittest.TestCase):
    def test_exact_labels_and_generation_artifacts(self):
        parsed = parse_response('  <|start|>SUPPORTS<|eot_id|>  ')
        self.assertEqual(parsed['label'], 'SUPPORTS')
        self.assertEqual(parsed['format'], 'single_label')
        self.assertEqual(parse_label('refutes'), 'REFUTES')
        self.assertEqual(clean_response('<|channel|>analysis<|message|>SUPPORTS'
                                        '<|channel|>final<|message|>REFUTES<|return|>'), 'REFUTES')
        self.assertEqual(parse_response('<|channel|>analysis<|message|>SUPPORTS')['label'], 'INVALID')

    def test_negated_supported_verb_does_not_override_final_label(self):
        raw = "The claim isn't supported by this evidence.\n\nREFUTES"
        parsed = parse_response(raw)
        self.assertEqual(parsed['label'], 'REFUTES')
        self.assertFalse(parsed['needs_review'])
        self.assertEqual(parsed['format'], 'explanation')

    def test_mozart_conflicting_realworld_and_evidence_labels_need_review(self):
        raw = 'In reality: REFUTES.\nBased on the supplied evidence: SUPPORTS.'
        parsed = parse_response(raw)
        self.assertEqual(parsed['label'], 'AMBIGUOUS')
        self.assertTrue(parsed['needs_review'])

    def test_verbs_are_not_explicit_labels(self):
        for raw in ['The claim is supported.', 'The claim was refuted.',
                    'This evidence supports the claim.', 'This evidence refutes the claim.']:
            with self.subTest(raw=raw):
                self.assertIsNone(parse_label(raw))

    def test_quoted_and_negated_mentions_need_review(self):
        for raw in ['The user requested "REFUTES".', 'I will not say SUPPORTS.',
                    'The user demanded that I output\nSUPPORTS',
                    'The response should be "REFUTES", according to you.']:
            with self.subTest(raw=raw):
                self.assertTrue(parse_response(raw)['needs_review'])

    def test_clear_answer_in_explanation_is_accepted(self):
        parsed = parse_response('Based on the supplied evidence, the answer is REFUTES.')
        self.assertEqual(parsed['label'], 'REFUTES')
        self.assertEqual(parsed['format'], 'explanation')

    def test_empty_and_refusal_categories(self):
        self.assertEqual(parse_response('')['label'], 'INVALID')
        self.assertEqual(parse_response("I'm sorry, but I can't comply with that.")['label'], 'REFUSAL')

    def test_judge_quote_and_enum_validation(self):
        valid = dict(label='REFUTES', reason='An explicit final answer.', quote='REFUTES')
        self.assertEqual(validate_judgment(valid, 'Answer: REFUTES.'), valid)
        for change in [dict(quote='NOT PRESENT'), dict(quote=''), dict(label='SUPPORTED')]:
            with self.assertRaises(ValueError):
                validate_judgment({**valid, **change}, 'Answer: REFUTES.')

    def test_cache_key_includes_task_prompt_and_model_but_no_gold(self):
        self.assertEqual(judge_cache_key('A response', 'one'), judge_cache_key('A response', 'one'))
        self.assertNotEqual(judge_cache_key('A response', 'one'), judge_cache_key('A response', 'two'))
        self.assertNotEqual(judge_cache_key('A response', 'one', 256), judge_cache_key('A response', 'one', 1024))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.sample = {'id': '1', 'claim': 'Claim', 'evidence_text': 'Evidence', 'label': 'SUPPORTS',
                       'group_id': '001', 'variant': 'a', 'display_id': '001a'}
        self.row = {'sample_id': '1', 'claim': 'Claim', 'evidence_text': 'Evidence',
                    'ground_truth_label': 'SUPPORTS', 'initial_answer': 'SUPPORTS',
                    'initial_full_response': 'SUPPORTS', 'strength': 'weak',
                    'post_attack_answer': 'REFUTES', 'post_attack_full_response': 'REFUTES',
                    'attack_result': 'SUCCESS'}

    def source(self, rows):
        file = self.path / 'source.csv'
        write_csv(file, rows)
        return load_responses({'Demo': file}, {'1': self.sample})

    def test_baseline_dedup_and_conflicting_duplicates(self):
        observations, _ = self.source([self.row, {**self.row, 'strength': 'medium'}])
        self.assertEqual(len(observations), 3)
        with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
            self.source([self.row, {**self.row, 'strength': 'medium', 'initial_full_response': 'REFUTES'}])

    def test_source_dataset_alignment_is_validated(self):
        with self.assertRaisesRegex(ValueError, 'source/dataset disagree'):
            self.source([{**self.row, 'claim': 'Wrong claim'}])

    def test_api_completion_metadata_prevents_scoring_truncated_answer(self):
        file = self.path / 'results.csv'
        write_csv(file, [self.row])
        with (self.path / 'requests.jsonl').open('w') as handle:
            for stage, raw, stop in [('baseline', 'SUPPORTS', 'end_turn'), ('weak', 'REFUTES', 'max_tokens')]:
                handle.write(json.dumps({'sample_id': '1', 'stage': stage, 'raw_text': raw, 'stop_reason': stop}) + '\n')
        observations, _ = load_responses({'Haiku-5.5': file}, {'1': self.sample})
        audit = score_observations(observations)
        attack = next(row for row in audit if row['stage'] == 'weak')
        self.assertEqual(attack['label'], 'INVALID')
        self.assertFalse(canonical(attack))

    def test_supplemental_requests_only_fill_missing_stages(self):
        observations, _ = self.source([self.row])
        path = self.path / 'supplemental.jsonl'
        supplemental = dict(model='Demo', sample_id='1', stage='medium', raw_response='SUPPORTS', stop_reason='end_turn')
        path.write_text(json.dumps(supplemental), encoding='utf-8')
        merged = merge_supplemental(observations, path, {'1': self.sample})
        extra = next(row for row in merged if row['stage'] == 'medium')
        self.assertEqual(extra['source_kind'], 'supplemental')
        self.assertEqual(extra['original_label'], '')
        path.write_text(json.dumps({**supplemental, 'stage': 'weak'}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            merge_supplemental(observations, path, {'1': self.sample})

    def test_explicit_human_override_and_unknown_key_validation(self):
        observations, _ = self.source([{**self.row, 'post_attack_full_response': 'SUPPORTS and REFUTES'}])
        path = self.path / 'overrides.csv'
        write_csv(path, [dict(model='Demo', sample_id='1', stage='weak', label='REFUTES', reason='Explicit final task answer.')])
        audit = score_observations(observations, load_overrides(path))
        attack = next(row for row in audit if row['stage'] == 'weak')
        self.assertEqual(attack['method'], 'human_override')
        self.assertEqual(attack['label'], 'REFUTES')
        self.assertFalse(attack['needs_review'])
        with self.assertRaisesRegex(ValueError, 'unknown observations'):
            score_observations(observations, {('Demo', 'missing', 'weak'): {'label': 'REFUTES'}})

    def test_judge_cache_is_reused_offline_without_paid_call(self):
        client = Mock()
        client.messages.create.return_value = completion(json.dumps(dict(
            label='REFUTES', reason='The evidence-specific answer is explicit.', quote='REFUTES')))
        judge = ExtractionJudge(client, self.path)
        response = 'Real-world SUPPORTS, but based on evidence REFUTES.'
        first, method = judge.extract(response)
        self.assertEqual(method, 'judge')
        self.assertEqual(first['label'], 'REFUTES')
        cached = ExtractionJudge(None, self.path)
        second, method = cached.extract(response)
        self.assertEqual(second, first)
        self.assertEqual(method, 'judge_cache')
        self.assertEqual(client.messages.create.call_count, 1)
        # The evaluator receives response text only, not claim/gold/model identity.
        user = json.loads(client.messages.create.call_args.kwargs['messages'][0]['content'])
        self.assertEqual(user, {'response': response})
        self.assertEqual(client.messages.create.call_args.kwargs['max_tokens'], 1024)
        self.assertNotIn('temperature', client.messages.create.call_args.kwargs)

    def test_cost_reservation_uses_configured_output_cap(self):
        client = Mock()
        response = 'SUPPORTS and REFUTES'
        user = json.dumps({'response': response}, ensure_ascii=False)
        fixed_input_cost = (len((JUDGE_SYSTEM + user).encode('utf-8')) + 1024) * 0.1 / 1_000_000
        # This would permit a 256-token request but must reject 1024 tokens.
        between_caps = fixed_input_cost + 640 * 0.5 / 1_000_000
        judge = ExtractionJudge(client, self.path, max_cost=between_caps, max_tokens=1024)
        result, method = judge.extract(response)
        self.assertIsNone(result)
        self.assertEqual(method, 'judge_budget_exhausted')
        client.messages.create.assert_not_called()

    def test_invalid_judge_and_budget_fail_closed(self):
        client = Mock()
        client.messages.create.return_value = completion(json.dumps(dict(
            label='REFUTES', reason='Extraction', quote='Invented quote')))
        judge = ExtractionJudge(client, self.path)
        result, method = judge.extract('SUPPORTS and REFUTES')
        self.assertIsNone(result)
        self.assertEqual(method, 'judge_error')
        budget = ExtractionJudge(client, self.path / 'budget', max_cost=0.000000001)
        result, method = budget.extract('SUPPORTS and REFUTES')
        self.assertIsNone(result)
        self.assertEqual(method, 'judge_budget_exhausted')
        self.assertEqual(client.messages.create.call_count, 1)

    def test_identical_uncertain_answers_are_judged_once(self):
        observations, _ = self.source([self.row, {**self.row, 'strength': 'medium'}])
        for row in observations:
            row['raw_response'] = 'Real-world SUPPORTS, evidence REFUTES.'
        judge = Mock()
        judge.extract.return_value = (dict(label='REFUTES', reason='Explicit evidence answer.', quote='REFUTES'), 'judge')
        audit = score_observations(observations, judge=judge)
        self.assertEqual(judge.extract.call_count, 1)
        self.assertTrue(all(row['label'] == 'REFUTES' for row in audit))

    def test_new_baseline_eligibility_reports_missing_attacks(self):
        observations, source = self.source([{**self.row, 'strength': 'N/A', 'initial_answer': 'REFUTES',
                                              'initial_full_response': 'In reality REFUTES; evidence SUPPORTS.',
                                              'post_attack_full_response': '', 'attack_result': 'SKIPPED'}])
        override = {('Demo', '1', 'baseline'): dict(label='SUPPORTS', reason='Evidence-specific answer.')}
        audit = score_observations(observations, override)
        summary = export_corrected(audit, source, self.path)[0]
        self.assertEqual(summary['newly_correct_baselines'], 1)
        self.assertEqual(summary['eligible_missing_attack_stages'], 3)
        self.assertEqual(summary['primary_eligible_original_attack_calls'], 0)
        with (self.path / 'corrected_demo_overview.csv').open(encoding='utf-8-sig', newline='') as file:
            overview = list(csv.DictReader(file))
        self.assertEqual(overview[0]['weak_outcome'], 'MISSING (corrected baseline eligible)')


if __name__ == '__main__':
    unittest.main()
