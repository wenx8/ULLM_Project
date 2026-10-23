"""Conservative extraction of a saved answer, independent of its gold label.

This module never solves the classification task. Ordinary verbs such as
"supported" and "refuted" are deliberately not classification labels.
"""
import re

LABELS = frozenset({'SUPPORTS', 'REFUTES'})
DECISIONS = LABELS | {'AMBIGUOUS', 'REFUSAL', 'INVALID'}
LABEL_RE = re.compile(r'\b(SUPPORTS|REFUTES)\b')
TOKEN_RE = re.compile(r'<\|[^|>]*\|>|</?(?:s|eos|bos)>')
REFUSAL_RE = re.compile(
    r"\b(?:I\s+(?:cannot|can't|can’t|won't|will not|am unable to)|"
    r"unable to (?:answer|comply)|cannot (?:answer|comply)|"
    r"sorry[, ]+(?:but\s+)?I\s+(?:cannot|can't|can’t))\b", re.I)
CAUTION_RE = re.compile(
    r"\b(?:not|never|isn['’]?t|aren['’]?t|doesn['’]?t|don['’]?t|"
    r"cannot|can['’]?t|won['’]?t|refuse|requested|requesting|asked|"
    r"suggested|suggesting|instructed|instruction|hypothetical|demanded|"
    r"rather than|instead of|would be|could be|might be|should be)\b", re.I)
META_RE = re.compile(
    r"\b(?:requested|requesting|asked|suggested|suggesting|instructed|instruction|"
    r"hypothetical|demanded|rather than|instead of|would be|could be|might be|should be|"
    r"(?:not|never|cannot|can['’]?t)\s+(?:say|output|answer|respond))\b", re.I)


def clean_response(raw):
    """Remove known generation artifacts, preserving all answer wording.

    GPT-OSS final-channel extraction matches the existing generation routine.
    Analysis-only generations have no saved final answer and remain empty.
    """
    value = '' if raw is None else str(raw)
    marker = '<|channel|>final<|message|>'
    if marker in value:
        value = value.rsplit(marker, 1)[1]
    elif '<|channel|>' in value:
        return ''
    return TOKEN_RE.sub('', value).strip()


def _decision(clean, label, format_, method, reason, quote='', needs_review=False):
    return {'label': label, 'format': format_, 'method': method,
            'reason': reason, 'quote': quote, 'needs_review': needs_review,
            'clean_response': clean}


def parse_response(raw):
    """Return an auditable decision without considering the expected label.

    Exact words are accepted directly. A longer response with one distinct
    canonical label is accepted only when that label is not used in a quoted,
    negated, hypothetical, or instruction-reporting context. Other responses
    need an extraction-only judge or human review; first-match scoring is
    never used.
    """
    clean = clean_response(raw)
    if not clean:
        return _decision(clean, 'INVALID', 'invalid', 'empty',
                         'No saved final-answer text.', needs_review=True)
    if clean.upper() in LABELS:
        return _decision(clean, clean.upper(), 'single_label', 'exact',
                         'Complete answer is one canonical label.', clean)
    matches = list(LABEL_RE.finditer(clean))
    distinct = {match.group().upper() for match in matches}
    if len(distinct) == 1:
        # A final explicit label on its own line is a clear task answer if the
        # response contains no other canonical label. Negations in preceding
        # prose ("isn't supported") do not turn into labels.
        last_line = clean.splitlines()[-1].strip()
        unwrapped = last_line.strip('`*_ \t').rstrip('.!').strip()
        if (unwrapped.upper() in LABELS and len(clean.splitlines()) > 1
                and not META_RE.search(clean[:-len(last_line)])):
            return _decision(clean, unwrapped.upper(), 'explanation', 'final_line',
                             'Only canonical label is an explicit final line.', last_line)
        risky = bool(CAUTION_RE.search(clean))
        for match in matches:
            start = max(clean.rfind('\n', 0, match.start()),
                        clean.rfind('.', 0, match.start()),
                        clean.rfind('!', 0, match.start()),
                        clean.rfind('?', 0, match.start())) + 1
            end = min([p for p in (clean.find('\n', match.end()),
                                    clean.find('.', match.end()),
                                    clean.find('!', match.end()),
                                    clean.find('?', match.end())) if p >= 0] or [len(clean)])
            sentence = clean[start:end]
            before = clean[max(start, match.start() - 2):match.start()]
            after = clean[match.end():min(end, match.end() + 2)]
            quoted = any(char in before + after for char in '\"\'“”‘’')
            if CAUTION_RE.search(sentence) or quoted:
                risky = True
                break
        if not risky:
            return _decision(clean, next(iter(distinct)), 'explanation', 'canonical_unique',
                             'One unopposed explicit canonical answer label.', matches[-1].group())
    if not matches and REFUSAL_RE.search(clean):
        return _decision(clean, 'REFUSAL', 'refusal', 'refusal_pattern',
                         'Explicit refusal without an answer label.', needs_review=True)
    return _decision(clean, 'AMBIGUOUS', 'ambiguous', 'review_required',
                     ('Both canonical labels occur; task answer needs interpretation.'
                      if len(distinct) > 1 else
                      'Answer label is quoted, negated, hypothetical, or not explicit.'),
                     needs_review=True)


def parse_label(raw):
    """Compatibility helper: return a safely extracted label, or None."""
    result = parse_response(raw)
    return result['label'] if result['label'] in LABELS and not result['needs_review'] else None
