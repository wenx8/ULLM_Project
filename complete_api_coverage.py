"""Collect missing API follow-ups after baseline rescoring, without replacing old runs."""
import argparse
import csv
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from llm_test import ATTACK_PROMPTS, build_baseline_prompt

BASE = Path(__file__).resolve().parent
MODELS = {'Haiku-5.5': ('claude-haiku-5-5', .1, .5),
          'Sonnet-5.5': ('claude-sonnet-5-5', 2., 10.)}


def find_missing(rows, completed=()):
    """Only correct, resolved API baselines qualify. Never regenerate a baseline."""
    present = {(r['model'], r['sample_id'], r['stage']) for r in rows} | set(completed)
    jobs = []
    for row in rows:
        if (row['stage'] != 'baseline' or row['model'] not in MODELS or
                row['label'] != row['ground_truth_label'] or
                str(row.get('needs_review', '')).lower() in ('true', '1', 'yes')):
            continue
        for strength in ATTACK_PROMPTS:
            if (row['model'], row['sample_id'], strength) not in present:
                jobs.append((row, strength))
    return jobs


class Budget:
    def __init__(self, limit, spent=0):
        self.limit, self.spent, self.reserved = limit, spent, 0.
        self.lock = threading.Lock()

    def reserve(self, amount):
        with self.lock:
            if self.spent + self.reserved + amount > self.limit:
                raise RuntimeError('Supplemental-call budget reached; saved calls can be resumed.')
            self.reserved += amount

    def settle(self, reserved, actual):
        with self.lock:
            self.reserved -= reserved
            self.spent += actual


def run_job(client, row, strength, budget, output, lock):
    model, input_rate, output_rate = MODELS[row['model']]
    target = 'REFUTES' if row['ground_truth_label'] == 'SUPPORTS' else 'SUPPORTS'
    prompt = ATTACK_PROMPTS[strength].format(target=target)
    sample = {'claim': row['claim'], 'evidence_text': row['evidence_text']}
    messages = [{'role': 'user', 'content': build_baseline_prompt(sample)},
                {'role': 'assistant', 'content': row['raw_response']},
                {'role': 'user', 'content': prompt}]
    # UTF-8 bytes plus generous framing allowance bounds input tokenization for
    # these short text-only calls. Output is capped, and SDK retries are disabled.
    upper_input = len(json.dumps(messages, ensure_ascii=False).encode('utf-8')) + 4096
    reserve = (upper_input * input_rate + 2048 * output_rate) / 1e6
    budget.reserve(reserve)
    try:
        response = client.messages.create(model=model, messages=messages, max_tokens=2048,
                                          output_config={'effort': 'low'})
    except Exception:
        # Conservatively retain the reservation on unknown server-side outcomes.
        raise
    usage = response.usage.model_dump()
    actual = (usage.get('input_tokens', 0) * input_rate +
              usage.get('output_tokens', 0) * output_rate) / 1e6
    record = {'model': row['model'], 'model_id': response.model,
              'sample_id': row['sample_id'], 'stage': strength,
              'raw_response': ''.join(b.text for b in response.content if b.type == 'text').strip(),
              'stop_reason': response.stop_reason, 'usage': usage,
              'request_id': getattr(response, '_request_id', None),
              'time': datetime.now(timezone.utc).isoformat(),
              'history_replayed_from_visible_text': True, 'estimated_cost_usd': actual}
    with lock:
        with output.open('a', encoding='utf-8') as file:
            file.write(json.dumps(record, ensure_ascii=False) + '\n')
    budget.settle(reserve, actual)
    return row['model'], row['sample_id'], strength


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=BASE / 'results/review/response_audit.csv')
    parser.add_argument('--output', type=Path, default=BASE / 'results/review/supplemental_requests.jsonl')
    parser.add_argument('--max-cost', type=float, default=.50)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.max_cost <= 0 or args.workers < 1:
        parser.error('Cost and workers must be positive.')
    with args.audit.open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    previous = ([json.loads(line) for line in args.output.read_text(encoding='utf-8').splitlines() if line.strip()]
                if args.output.exists() else [])
    completed = {(r['model'], r['sample_id'], r['stage']) for r in previous}
    jobs = find_missing(rows, completed)
    print(f'{len(jobs)} missing API attacks; max supplemental budget ${args.max_cost:.2f}.')
    if args.dry_run or not jobs:
        return
    if not os.environ.get('ANTHROPIC_API_KEY'):
        raise SystemExit('Set ANTHROPIC_API_KEY before collecting missing API follow-ups.')
    import anthropic
    args.output.parent.mkdir(parents=True, exist_ok=True)
    budget = Budget(args.max_cost, sum(r.get('estimated_cost_usd', 0) for r in previous))
    lock = threading.Lock()
    with anthropic.Anthropic(max_retries=0, timeout=120.) as client, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, client, row, stage, budget, args.output, lock) for row, stage in jobs]
        for future in as_completed(futures):
            print('Saved:', *future.result(), flush=True)
    print(f'Logged supplemental cost: ${budget.spent:.6f}')


if __name__ == '__main__':
    main()
