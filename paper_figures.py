r"""Generate two simple bar charts from audited CSVs. No API calls."""
import csv
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(BASE / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

MODELS = ('GPT-OSS-20B', 'Llama-3.1-8B', 'Haiku-5.5', 'Sonnet-5.5')
ANALYSIS = BASE / 'results/analysis'
OUTPUT = BASE / 'results/paper_figures'
COLORS = ('#3979AC', '#E29B35', '#5B9B83')


def read_table(name):
    with (ANALYSIS / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def bar_chart(series, title, ylabel, stem):
    fig, ax = plt.subplots(figsize=(9, 4.6), layout='constrained')
    width = .23 if len(series) == 3 else .30
    for index, (label, values) in enumerate(series):
        positions = [x + (index - (len(series) - 1) / 2) * width
                     for x in range(len(MODELS))]
        bars = ax.bar(positions, values, width=width * .91,
                      color=COLORS[index], label=label, zorder=3)
        ax.bar_label(bars, labels=[f'{value:.1f}%' for value in values],
                     padding=4, fontsize=10)
    ax.set_xticks(range(len(MODELS)), MODELS)
    ax.set_ylim(0, 113)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left', fontsize=14, fontweight='bold', pad=15)
    ax.legend(frameon=False, ncol=len(series), loc='upper left')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#aaaaaa')
    ax.tick_params(axis='both', length=0, pad=7)
    ax.grid(axis='y', color='#e8e8e8', linewidth=.7, zorder=0)
    ax.set_axisbelow(True)
    for extension in ('png', 'pdf'):
        buffer = BytesIO()
        fig.savefig(buffer, format=extension, dpi=300,
                    bbox_inches='tight', facecolor='white')
        (OUTPUT / f'{stem}.{extension}').write_bytes(buffer.getvalue())
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    names = ('aggregate_attacks.csv', 'baseline_by_evidence.csv')
    attacks = {(r['model'], r['strength']): r for r in read_table(names[0])
               if r['cohort'] == 'original' and r['direction'] == 'ALL'
               and r['strength'] in ('weak', 'medium', 'strong')}
    assert len(attacks) == 12
    series = []
    for strength in ('weak', 'medium', 'strong'):
        values = []
        for model in MODELS:
            row = attacks[model, strength]
            rate = int(row['flips']) / int(row['observed_attacks'])
            assert abs(rate - float(row['confirmed_asr'])) < 1e-12
            values.append(rate * 100)
        series.append((strength.capitalize(), values))
    bar_chart(series, 'How often did a conflicting follow-up flip a correct answer?',
              'Incorrect answer flips', 'figure1_attack_success_by_prompt')
    evidence = {(r['model'], r['evidence_type']): r for r in read_table(names[1])}
    series = []
    for kind in ('original', 'edited'):
        values = []
        for model in MODELS:
            row = evidence[model, kind]
            assert int(row['samples']) == 354
            assert int(row['supports_gold']) == int(row['refutes_gold']) == 177
            rate = int(row['correct']) / int(row['samples'])
            assert abs(rate - float(row['accuracy'])) < 1e-12
            values.append(rate * 100)
        series.append((kind.capitalize() + ' evidence', values))
    bar_chart(series, 'Every model was less accurate on edited evidence',
              'Baseline accuracy', 'figure2_original_vs_edited_evidence')
    sources = {str((ANALYSIS / name).relative_to(BASE)):
               hashlib.sha256((ANALYSIS / name).read_bytes()).hexdigest() for name in names}
    (OUTPUT / 'provenance.json').write_text(json.dumps({
        'sources_sha256': sources,
        'analysis_metadata': json.loads((ANALYSIS / 'metadata.json').read_text()),
        'note': 'Descriptive bar charts; error bars omitted for readability. '
                'Primary original-collection cohort, no supplemental attacks or API calls. '
                'Labels rounded to one decimal; CSVs retain full precision.'
    }, indent=2) + '\n', encoding='utf-8')
    print(f'Saved two simple charts as PDF and 300-dpi PNG in {OUTPUT}')


if __name__ == '__main__':
    main()
