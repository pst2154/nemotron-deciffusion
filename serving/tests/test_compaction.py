import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import compaction  # noqa: E402


class WordTokenizer:
    def __call__(self, texts, add_special_tokens=False):
        return {'input_ids': [t.split() for t in texts]}

    def encode(self, text, add_special_tokens=False):
        return text.split()


TOK = WordTokenizer()
FILLER = ' '.join(f'The sky was grey on day {i}.' for i in range(800))


def test_short_state_is_untouched():
    state = {'note': 'short'}
    assert compaction.compact_state(TOK, state, 'q', ['a'], 500) == (state, None)


def test_prose_keeps_relevant_sentence_in_budget():
    state = {'story': FILLER[:20000] + ' Mary went to the kitchen. ' + FILLER[20000:], 'n': 3}
    out, info = compaction.compact_state(TOK, state, 'Where is Mary?', ['kitchen', 'garden'], 100)
    assert 'Mary went to the kitchen.' in out['story']
    assert info['tokens_after'] <= 100 and out['n'] == 3


def test_trace_string_keeps_head_and_tail():
    traj = 'TASK: fix parser\n' + '\n'.join(f'step {i}: ran tests' for i in range(2000)) + '\nFINAL: submitted'
    out, info = compaction.compact_state(TOK, {'trajectory': traj}, 'Did the run resolve the task?', ['yes', 'no'], 300)
    text = out['trajectory']
    assert info['mode'] == 'head_tail'
    assert text.startswith('TASK: fix parser') and text.endswith('FINAL: submitted')
    assert 'lines omitted' in text and len(text.split()) <= 320


def test_trace_list_keeps_first_and_recent_items():
    msgs = [{'role': 'user', 'content': 'TASK fix bug'}] + [
        {'role': 'assistant', 'content': f'step {i} ' + 'x ' * 30} for i in range(200)]
    out, _ = compaction.compact_state(TOK, {'messages': msgs}, 'Escalate?', ['continue', 'escalate'], 500)
    kept = out['messages']
    assert kept[0] == msgs[0] and kept[-1] == msgs[-1]
    assert any(isinstance(x, str) and 'items omitted' in x for x in kept)
    assert len(kept) < len(msgs)


def test_trace_keys_are_whole_words():
    for key in ('trajectory', 'agent_log', 'messages', 'history', 'steps'):
        assert compaction._TRACE_KEY.search(key)
    for key in ('catalog', 'dialog', 'login', 'blog', 'technology'):
        assert not compaction._TRACE_KEY.search(key)


def test_trace_steps_are_kept_whole():
    steps = [f'### Step {i}\n[assistant] working on it\n[tool_output]\nok {i}' for i in range(1, 400)]
    out, _ = compaction.compact_state(TOK, {'trajectory': '\n\n'.join(steps)}, 'Did it work?', ['yes', 'no'], 400)
    blocks = out['trajectory'].split('\n\n')
    assert blocks[0] == steps[0] and blocks[-1] == steps[-1]
    assert any('earlier steps omitted' in b for b in blocks)
    assert all(b.startswith('### Step') or 'omitted' in b for b in blocks)
