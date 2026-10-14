"""Shared helpers for the finetune/ scripts (stdlib only; nothing here imports torch or transformers)."""
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP = REPO / 'serving' / 'app'

# Field names that usually mean the answer leaked into the state. Extend with --leak-fields.
DEFAULT_LEAK_FIELDS = ('label', 'labels', 'answer', 'answers', 'gold', 'target', 'expected', 'ground_truth',
                       'groundtruth', 'outcome', 'correct', 'correct_answer', 'resolution', 'verdict',
                       'final_decision', 'passed', 'success')

NOUL_TRUE = {'true', 'yes', 'y', '1', 't'}
NOUL_FALSE = {'false', 'no', 'n', '0', 'f'}


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise SystemExit(f'{path}:{n}: invalid JSON ({e})')
    return rows


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def stable_hash(value, salt=''):
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256((salt + text).encode()).hexdigest()


def state_text(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def the_question(row):
    """Training rows carry exactly one question; return (key, question)."""
    qs = row['input']['questions']
    return next(iter(qs.items()))


def labels_of(question):
    """API labels in criteria order, as serving/app/scoring.py prepare_questions defines them."""
    kind = question['type']
    if kind == 'choice':
        return list(question.get('criteria') or {})
    if kind == 'noul':
        return ['false', 'true']
    return [str(i) for i in range(len(question.get('criteria') or []))]


def option_texts(question):
    kind = question['type']
    crit = question.get('criteria') or {}
    if kind == 'choice':
        return {k: state_text(v) for k, v in crit.items()}
    if kind == 'noul':
        return {'false': crit.get('false', 'No, the condition does not hold.'),
                'true': crit.get('true', 'Yes, the condition holds.')}
    return {str(i): state_text(v) for i, v in enumerate(crit)}


def gold_label(row):
    """The target exactly as train_pack.py prepare() normalises it."""
    kind = the_question(row)[1]['type']
    t = row['reference']['target']
    return str(t).lower() if kind == 'noul' else str(t)


def normalize_noul(value):
    v = str(value).strip().lower()
    if v in NOUL_TRUE:
        return 'true'
    if v in NOUL_FALSE:
        return 'false'
    return None


def find_keys(value, names, path=()):
    """Yield paths of dict keys in `value` whose lower-cased name is in `names`."""
    if isinstance(value, dict):
        for k, v in value.items():
            if str(k).lower() in names:
                yield path + (k,)
            yield from find_keys(v, names, path + (k,))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from find_keys(v, names, path + (i,))


_WORD = re.compile(r'\w+')


def ngrams(text, n):
    words = _WORD.findall(text.lower())
    return {' '.join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def import_scoring(app=None):
    """Import the serving prompt builder (serving/app/scoring.py) without importing torch or vLLM."""
    sys.path.insert(0, str(app or APP))
    import scoring  # noqa: E402
    return scoring


def load_tokenizer(checkpoint):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
