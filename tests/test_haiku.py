import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import haiku_test as runner


def response(text, stop='end_turn'):
    block = SimpleNamespace(type='text', text=text)
    block.model_dump = lambda **kwargs: {'type': 'text', 'text': text}
    return SimpleNamespace(
        content=[block], model='claude-haiku-5-5', stop_reason=stop,
        usage=SimpleNamespace(model_dump=lambda: {'input_tokens': 100, 'output_tokens': 5}),
    )


class HaikuTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.args = runner.parse_args([])
        self.sample = dict(id='1', claim='A claim', evidence_text='Evidence', label='SUPPORTS')
        self.client = Mock()

    def test_csv_reader_lock_is_retried_without_losing_results(self):
        from unittest.mock import patch
        target = self.path / 'results.csv'
        temporary = self.path / 'results.csv.tmp'
        temporary.write_text('new results', encoding='utf-8')
        original_replace = Path.replace
        attempts = []

        def transient_lock(path, destination):
            attempts.append(destination)
            if len(attempts) == 1:
                raise PermissionError('CSV is being read')
            return original_replace(path, destination)

        with patch.object(Path, 'replace', transient_lock), patch.object(runner.time, 'sleep'):
            exported = runner.replace_csv_export(temporary, target)
        self.assertEqual(exported, target)
        self.assertEqual(target.read_text(encoding='utf-8'), 'new results')

    def test_independent_attacks_and_usage(self):
        self.client.messages.create.side_effect = [response(x) for x in
                                                  ['SUPPORTS', 'REFUTES', 'SUPPORTS', 'REFUTES']]
        rows = runner.run_sample(self.client, self.args, self.sample, self.path / 'requests.jsonl')
        self.assertEqual([r['attack_result'] for r in rows], ['SUCCESS', 'FAILURE', 'SUCCESS'])
        calls = self.client.messages.create.call_args_list
        for call in calls[1:]:
            self.assertEqual(len(call.kwargs['messages']), 3)
            self.assertEqual(call.kwargs['messages'][1]['content'], [{'type': 'text', 'text': 'SUPPORTS'}])
        self.assertEqual(len((self.path / 'requests.jsonl').read_text().splitlines()), 4)

    def test_incorrect_baseline_skips_attacks(self):
        self.client.messages.create.return_value = response('REFUTES')
        rows = runner.run_sample(self.client, self.args, self.sample, self.path / 'requests.jsonl')
        self.assertEqual(self.client.messages.create.call_count, 1)
        self.assertEqual(rows[0]['attack_result'], 'SKIPPED (initial answer incorrect)')

    def test_all_dataset_families_and_stable_ids(self):
        samples = [json.loads(line) for line in
                   (runner.BASE / 'data/processed/pilot_fever_symmetric.jsonl').read_text(encoding='utf-8').splitlines()]
        grouped = runner.assign_groups(samples)
        self.assertEqual(len({s['group_id'] for s in grouped}), 177)
        self.assertEqual([s['display_id'] for s in grouped[:4]], ['001a', '001b', '001c', '001d'])
        reverse = runner.assign_groups(list(reversed(samples)))
        self.assertEqual([(s['id'], s['display_id']) for s in grouped],
                         [(s['id'], s['display_id']) for s in reverse])

    def test_csv_overview_and_request_outcomes(self):
        self.sample.update(group_id='001', variant='a', display_id='001a')
        self.client.messages.create.side_effect = [response(x) for x in
                                                  ['SUPPORTS', 'REFUTES', 'SUPPORTS', 'REFUTES']]
        rows = runner.run_sample(self.client, self.args, self.sample, self.path / 'requests.jsonl')
        runner.export_csv({'completed': {'1': rows}}, self.path)
        with (self.path / 'overview.csv').open(encoding='utf-8-sig', newline='') as file:
            overview = list(csv.DictReader(file))
        self.assertEqual(len(overview), 1)
        self.assertEqual(overview[0]['display_id'], '001a')
        self.assertEqual(overview[0]['initial_answer'], 'SUPPORTS')
        self.assertEqual(overview[0]['weak_outcome'], 'SUCCESS')
        with (self.path / 'requests.csv').open(encoding='utf-8-sig', newline='') as file:
            requests = list(csv.DictReader(file))
        self.assertEqual([r['outcome'] for r in requests], ['CORRECT', 'SUCCESS', 'FAILURE', 'SUCCESS'])
        self.assertEqual(requests[0]['input_tokens'], '100')

    def test_truncated_label_is_invalid(self):
        self.client.messages.create.return_value = response('SUPPORTS', 'max_tokens')
        rows = runner.run_sample(self.client, self.args, self.sample, self.path / 'requests.jsonl')
        self.assertEqual(rows[0]['initial_answer'], 'INVALID')
        self.assertEqual(self.client.messages.create.call_count, 1)

    def test_canonical_mode_ignores_supported_verb(self):
        self.sample['label'] = 'SUPPORTS'
        self.client.messages.create.return_value = response("The claim isn't supported.\n\nREFUTES")
        rows = runner.run_sample(self.client, self.args, self.sample, self.path / 'requests.jsonl')
        self.assertEqual(rows[0]['initial_answer'], 'REFUTES')
        self.assertEqual(self.client.messages.create.call_count, 1)

    def test_legacy_mode_is_explicit_and_preserves_historical_behavior(self):
        self.args.scoring = 'legacy'
        self.client.messages.create.return_value = response("The claim isn't supported.\n\nREFUTES")
        rows = runner.run_sample(self.client, self.args, self.sample, self.path / 'requests.jsonl')
        self.assertEqual(rows[0]['initial_answer'], 'SUPPORTS')
        self.assertEqual(self.client.messages.create.call_count, 4)

    def test_concurrent_samples_preserve_all_requests_and_checkpoints(self):
        import sys
        from unittest.mock import patch
        dataset = self.path / 'parallel.jsonl'
        dataset.write_text('\n'.join(json.dumps({**self.sample, 'id': str(i)}) for i in range(1, 9)), encoding='utf-8')
        self.client.messages.create.side_effect = lambda **kwargs: response('REFUTES')
        context_client = Mock()
        context_client.__enter__ = Mock(return_value=self.client)
        context_client.__exit__ = Mock(return_value=False)
        module = SimpleNamespace(Anthropic=Mock(return_value=context_client))
        with patch.dict(sys.modules, {'anthropic': module}), patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'fake'}):
            runner.main(['--data', str(dataset), '--output-dir', str(self.path), '--workers', '4'])
        saved = json.loads((self.path / 'checkpoint.json').read_text(encoding='utf-8'))
        self.assertEqual(len(saved['completed']), 8)
        with (self.path / 'requests.csv').open(encoding='utf-8-sig', newline='') as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({row['sample_id'] for row in rows}), 8)

    def test_resume_without_sdk_or_credentials_and_config_guard(self):
        dataset = self.path / 'data.jsonl'
        dataset.write_text(json.dumps(self.sample), encoding='utf-8')
        args = ['--data', str(dataset), '--output-dir', str(self.path)]
        # Capture the same run configuration using a mocked client module.
        import sys
        from unittest.mock import patch
        self.client.messages.create.return_value = response('REFUTES')
        context_client = Mock()
        context_client.__enter__ = Mock(return_value=self.client)
        context_client.__exit__ = Mock(return_value=False)
        module = SimpleNamespace(Anthropic=Mock(return_value=context_client))
        with patch.dict(sys.modules, {'anthropic': module}), patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'fake'}):
            runner.main(args)
        self.client.reset_mock()
        runner.main(args)
        self.client.messages.create.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'settings/data changed'):
            runner.main(args + ['--effort', 'medium'])


if __name__ == '__main__':
    unittest.main()
