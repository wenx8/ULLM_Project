"""Run the existing three-strength experiment through the Anthropic API."""
import argparse
import csv
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path

from llm_test import ATTACK_PROMPTS, FIELDS, LABELS, build_baseline_prompt, legacy_normalize_answer
from scoring import parse_label

BASE = Path(__file__).resolve().parent
REQUEST_LOG_LOCK = threading.Lock()
WARNED_EXPORTS = set()
GROUP_FIELDS = ['group_id', 'variant', 'display_id']
REQUEST_FIELDS = GROUP_FIELDS + [
    'sample_id', 'stage', 'claim', 'evidence_text', 'ground_truth_label',
    'prompt', 'baseline_answer', 'answer', 'outcome', 'raw_text',
    'input_tokens', 'output_tokens', 'model', 'stop_reason', 'time', 'request_id',
]


def replace_csv_export(temporary, target):
    """Readers such as Excel can briefly deny file replacement on Windows."""
    for attempt in range(5):
        try:
            temporary.replace(target)
            return target
        except PermissionError:
            time.sleep(.1 * (2 ** attempt))
    alternate = target.with_name(target.stem + '.latest' + target.suffix)
    try:
        temporary.replace(alternate)
    except PermissionError:
        alternate = target.with_name(f'{target.stem}.{time.time_ns()}{target.suffix}')
        temporary.replace(alternate)
    if target not in WARNED_EXPORTS:
        print(f'CSV is open in another application; latest export saved to {alternate}', flush=True)
        WARNED_EXPORTS.add(target)
    return alternate


def assign_groups(samples):
    """Recognize original IDs and their 0000002/3/4 symmetric variants."""
    by_id = {str(sample['id']): sample for sample in samples}
    families = {}
    for original_id, sample in by_id.items():
        root = re.sub(r'000000[234]$', '', original_id)
        if root not in by_id:
            root = original_id  # Do not infer a family from a missing original.
        families.setdefault(root, []).append(sample)
    for number, root in enumerate(sorted(families, key=lambda x: (len(x), x)), 1):
        members = sorted(families[root], key=lambda s: (len(str(s['id'])), str(s['id'])))
        expected = {root, *(root + '000000' + str(i) for i in (2, 3, 4))}
        if len(members) > 1 and {str(s['id']) for s in members} != expected:
            raise ValueError(f'Incomplete symmetric family: {root}')
        if len(members) == 4:
            if (members[0]['claim'] != members[1]['claim'] or
                    members[2]['claim'] != members[3]['claim'] or
                    members[0]['evidence_text'] != members[2]['evidence_text'] or
                    members[1]['evidence_text'] != members[3]['evidence_text'] or
                    members[0]['label'] == members[1]['label'] or
                    members[2]['label'] == members[3]['label'] or
                    members[0]['label'] != members[3]['label']):
                raise ValueError(f'Unexpected symmetric relationships: {root}')
        for index, sample in enumerate(members):
            sample.update(group_id=f'{number:03}', variant=chr(97 + index),
                          display_id=f'{number:03}{chr(97 + index)}')
    return sorted(samples, key=lambda s: s['display_id'])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=BASE / 'data/processed/pilot_fever_symmetric.jsonl')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--model', default='claude-haiku-5-5')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--effort', choices=['low', 'medium', 'high'], default='low')
    parser.add_argument('--workers', type=int, default=1, help='Concurrent independent samples (default: 1)')
    parser.add_argument('--scoring', choices=['canonical', 'legacy'], default='canonical',
                        help='Use conservative labels for new runs; legacy resumes historical acquisition.')
    parser.add_argument('--dry-run', action='store_true', help='Validate data and show call limits without API access')
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = BASE / ('results/haiku_canonical' if args.scoring == 'canonical'
                                  else 'results/haiku_symmetric')
    if args.workers < 1 or args.max_tokens < 1 or (args.max_samples is not None and args.max_samples < 1):
        parser.error('Token and sample limits must be positive.')
    return args


def ask_haiku(client, args, messages, log_path, sample, stage, baseline_answer=None):
    response = client.messages.create(
        model=args.model,
        max_tokens=args.max_tokens,
        output_config={'effort': args.effort},
        messages=messages,
    )
    raw = ''.join(block.text for block in response.content if block.type == 'text').strip()
    extractor = legacy_normalize_answer if args.scoring == 'legacy' else parse_label
    answer = extractor(raw) if response.stop_reason == 'end_turn' else None
    outcome = ('INVALID' if answer is None else
               ('CORRECT' if answer == sample['label'] else 'INCORRECT') if stage == 'baseline'
               else ('FAILURE' if answer == sample['label'] else 'SUCCESS'))
    record = {
        'time': datetime.now(timezone.utc).isoformat(),
        'sample_id': str(sample['id']), 'stage': stage,
        **{key: sample.get(key, '') for key in GROUP_FIELDS},
        'claim': sample['claim'], 'evidence_text': sample['evidence_text'],
        'ground_truth_label': sample['label'], 'prompt': messages[-1]['content'],
        'baseline_answer': baseline_answer or '', 'answer': answer or 'INVALID',
        'outcome': outcome,
        'model': response.model, 'stop_reason': response.stop_reason,
        'usage': response.usage.model_dump(), 'raw_text': raw,
        'request_id': getattr(response, '_request_id', None),
        'scoring_policy': args.scoring,
    }
    with REQUEST_LOG_LOCK:
        with log_path.open('a', encoding='utf-8') as file:
            file.write(json.dumps(record, ensure_ascii=False) + '\n')
        export_requests(log_path)
    # Never score a truncated answer, even if it happens to contain a label.
    answer = extractor(raw) if response.stop_reason == 'end_turn' else None
    # Preserve the complete assistant content, including signed thinking blocks.
    content = [block.model_dump(exclude_none=True) for block in response.content]
    return raw, answer, content


def run_sample(client, args, sample, log_path):
    truth = sample['label']
    prompt = build_baseline_prompt(sample)
    messages = [{'role': 'user', 'content': prompt}]
    initial_raw, initial_answer, content = ask_haiku(
        client, args, messages, log_path, sample, 'baseline'
    )
    attacks = ATTACK_PROMPTS.items() if initial_answer == truth else [('N/A', '')]
    opposite = 'REFUTES' if truth == 'SUPPORTS' else 'SUPPORTS'
    rows = []
    for strength, template in attacks:
        attack_prompt = template.format(target=opposite)
        post_raw, post_answer = '', None
        if attack_prompt:
            # Each strength starts independently from the SAME baseline response.
            attack_messages = messages + [
                {'role': 'assistant', 'content': content},
                {'role': 'user', 'content': attack_prompt},
            ]
            post_raw, post_answer, _ = ask_haiku(
                client, args, attack_messages, log_path, sample, strength, initial_answer
            )
            result = ('INVALID (post-attack answer)' if post_answer is None else
                      'SUCCESS' if post_answer == opposite else 'FAILURE')
        else:
            result = ('SKIPPED (invalid initial answer)' if initial_answer is None
                      else 'SKIPPED (initial answer incorrect)')
        rows.append(dict(zip(FIELDS, [
            sample['id'], sample['claim'], sample['evidence_text'],
            sample.get('evidence_length_words', len(sample['evidence_text'].split())),
            truth, prompt, initial_answer or 'INVALID', strength, attack_prompt,
            post_answer or ('INVALID' if attack_prompt else 'N/A'), result,
            initial_raw, post_raw,
        ])))
        rows[-1].update({key: sample.get(key, '') for key in GROUP_FIELDS})
    return rows


def export_requests(log_path):
    records = [json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    stages = {'baseline': 0, 'weak': 1, 'medium': 2, 'strong': 3}
    records.sort(key=lambda r: (r.get('display_id', r['sample_id']), stages.get(r['stage'], 4), r['time']))
    target = log_path.with_suffix('.csv')
    temporary = target.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=REQUEST_FIELDS, extrasaction='ignore')
        writer.writeheader()
        for record in records:
            writer.writerow({**record, 'input_tokens': record['usage'].get('input_tokens', 0),
                             'output_tokens': record['usage'].get('output_tokens', 0)})
    replace_csv_export(temporary, target)


def export_csv(state, output_dir):
    temporary = output_dir / 'results.csv.tmp'
    with temporary.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=GROUP_FIELDS + FIELDS)
        writer.writeheader()
        for rows in sorted(state['completed'].values(), key=lambda rows: rows[0].get('display_id', str(rows[0]['sample_id']))):
            writer.writerows(rows)
    replace_csv_export(temporary, output_dir / 'results.csv')
    columns = GROUP_FIELDS + ['sample_id', 'claim', 'evidence_text', 'ground_truth_label', 'initial_answer']
    columns += [f'{strength}_{suffix}' for strength in ATTACK_PROMPTS for suffix in ('answer', 'outcome')]
    temporary = output_dir / 'overview.csv.tmp'
    with temporary.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for rows in sorted(state['completed'].values(), key=lambda rows: rows[0].get('display_id', str(rows[0]['sample_id']))):
            row = {key: rows[0].get(key, '') for key in columns[:8]}
            for strength in ATTACK_PROMPTS:
                attack = next((r for r in rows if r['strength'] == strength), None)
                row[f'{strength}_answer'] = attack['post_attack_answer'] if attack else 'N/A'
                row[f'{strength}_outcome'] = attack['attack_result'] if attack else rows[0]['attack_result']
            writer.writerow(row)
    replace_csv_export(temporary, output_dir / 'overview.csv')


def main(argv=None):
    args = parse_args(argv)
    data_bytes = args.data.read_bytes()
    samples = [json.loads(line) for line in data_bytes.decode('utf-8-sig').splitlines() if line.strip()]
    ids = set()
    for sample in samples:
        for key in ('id', 'claim', 'evidence_text', 'label'):
            if key not in sample:
                raise ValueError(f'Missing dataset field: {key}')
        if sample['label'] not in LABELS or str(sample['id']) in ids:
            raise ValueError('Dataset contains an unexpected label or duplicate ID.')
        ids.add(str(sample['id']))
    samples = assign_groups(samples)
    if args.max_samples is not None:
        samples = samples[:args.max_samples]
    if not samples:
        raise ValueError('Dataset is empty.')
    config = {
        'model': args.model, 'effort': args.effort, 'max_tokens': args.max_tokens,
        'dataset_sha256': hashlib.sha256(data_bytes).hexdigest(),
        'attack_prompts': ATTACK_PROMPTS,
        'baseline_template': build_baseline_prompt({'claim': '{claim}', 'evidence_text': '{evidence}'}),
        'protocol_version': 2 if args.scoring == 'legacy' else 3,
    }
    if args.scoring != 'legacy':
        config['scoring_policy'] = args.scoring
    checkpoint = args.output_dir / 'checkpoint.json'
    state = {'config': config, 'completed': {}}
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding='utf-8'))
        if state['config'] != config:
            raise ValueError('Run settings/data changed. Choose a different --output-dir.')
    pending = [s for s in samples if str(s['id']) not in state['completed']]
    print(f'{args.model}: {len(samples)} samples, {len(pending)} pending; at most {4 * len(pending)} new calls.')
    print(f'Effort={args.effort}; max_tokens={args.max_tokens}; output={args.output_dir}')
    if args.dry_run:
        return
    if not pending:
        export_csv(state, args.output_dir)
        return
    if not os.environ.get('ANTHROPIC_API_KEY'):
        raise SystemExit('Set ANTHROPIC_API_KEY before running. See README.md.')
    try:
        import anthropic
    except ImportError:
        raise SystemExit('Install API dependencies: python -m pip install -r requirements-api.txt')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The SDK retries transient failures twice; authentication/billing errors stop the run.
    with anthropic.Anthropic(max_retries=2, timeout=120.0) as client, ThreadPoolExecutor(max_workers=args.workers) as pool:
        iterator = iter(pending)
        futures = {}

        def submit_next():
            sample = next(iterator, None)
            if sample is not None:
                future = pool.submit(run_sample, client, args, sample, args.output_dir / 'requests.jsonl')
                futures[future] = sample

        for _ in range(args.workers):
            submit_next()
        number = 0
        while futures:
            ready, _ = wait(futures, return_when=FIRST_COMPLETED)
            errors = []
            for future in ready:
                sample = futures.pop(future)
                try:
                    rows = future.result()
                except Exception as error:
                    errors.append(error)
                    continue
                number += 1
                save_sample(state, checkpoint, args.output_dir, sample, rows)
                print(f'{number}/{len(pending)} saved: {sample["id"]}', flush=True)
            if errors:
                for future in futures:
                    future.cancel()
                # Preserve other in-flight successes before propagating the error.
                for future, sample in futures.items():
                    if not future.cancelled():
                        try:
                            rows = future.result()
                        except Exception:
                            continue
                        save_sample(state, checkpoint, args.output_dir, sample, rows)
                raise errors[0]
            for _ in ready:
                submit_next()
    print(f'Results: {args.output_dir / "results.csv"}')


def save_sample(state, checkpoint, output_dir, sample, rows):
    state['completed'][str(sample['id'])] = rows
    temporary = checkpoint.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(checkpoint)
    export_csv(state, output_dir)


if __name__ == '__main__':
    main()
