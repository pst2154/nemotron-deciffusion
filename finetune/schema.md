# Row schema and request mapping

Every builder in `training/`, every trainer (`training/train_pack.py`, `finetune/train_lora.py`) and every
script in `finetune/` uses the same JSONL row: one line per decision, exactly one question per row.

## Training row

```json
{
  "id": "support:ticket-0042",
  "input": {
    "state": {"ticket": {"subject": "Charged twice", "body": "Two charges of $49 on Mar 3."}},
    "questions": {
      "decision": {
        "type": "choice",
        "instructions": "Which support queue should handle this ticket?",
        "criteria": {"billing": "Payments, invoices, refunds", "technical": "Bugs and errors",
                     "account_security": "Suspicious sign-ins, 2FA", "sales": "Upgrades and quotes"}
      }
    }
  },
  "reference": {"target": "billing", "source": "helpdesk_resolution"},
  "teacher_probabilities": {"billing": 0.9, "technical": 0.04, "account_security": 0.03, "sales": 0.03},
  "provenance": {"source_id": "ticket-0042"},
  "domain": "support-routing"
}
```

| field | required | meaning |
|---|---|---|
| `id` | yes | unique string |
| `input.state` | yes | any JSON value (object, list or string): the evidence. It is serialized into the prompt as is. |
| `input.questions` | yes | object with exactly one entry. The key name does not matter (`decision` by convention). |
| `…type` | yes | `choice`, `noul` or `score` |
| `…instructions` | yes | the question text |
| `…criteria` | per type | see below |
| `reference.target` | yes | the gold label (see below). `reference.source` is free text. |
| `teacher_probabilities` | no | soft target over **all** labels of the question, keyed by API label; normalized by the trainer. When present it replaces the one-hot target in the loss. |
| `provenance.source_id` | recommended | groups rows that share an underlying item (same document, same trajectory, same customer). Keep groups on one side of the train/eval split. |
| `domain` | recommended | used for reporting and for per-domain metrics |

Extra keys are ignored by the trainers.

### Question types

| type | `criteria` | labels | `reference.target` | `teacher_probabilities` keys |
|---|---|---|---|---|
| `choice` | object `{key: description}`, 2 to 501 entries. Keys are what the API returns, descriptions are what the model reads (strings or JSON values). | the keys | one key | the keys |
| `noul` | optional `{"true": "...", "false": "..."}` descriptions. Default: "Yes, the condition holds." / "No, the condition does not hold." | `false`, `true` | `"true"` or `"false"` (also accepts booleans; the trainer lower-cases) | `"false"`, `"true"` |
| `score` | list of level descriptions, **ordered lowest to highest** (level i = position i), at least 2 | `"0"` … `"n-1"` | the level index as a string, e.g. `"2"` | `"0"` … `"n-1"` |

`score` levels are ordinal: never shuffle the criteria list. The API returns the probability of each level and
the expected level.

### What the model actually sees

`serving/app/scoring.py prepare_questions` turns a row into the prompt, identically at train and serve time:

1. Options are reordered by a fixed rule (`order_options`: description length and word overlap with the state,
   then a cyclic shift of 2). Your criteria order therefore does not matter for the canonical prompt, and the
   label you get back is always your key, never a position.
2. Each option is given a single-token code (`A`…`Z`, `a`…`z`, `0`…`9`, then further single-token codes such
   as two-letter uppercase pairs; 501 in total, `candidate_codes`). Prompts with 62 or fewer options only use
   letters and digits.
3. The chat template is applied to a fixed system message and a JSON user turn
   `{"state": ..., "question": instructions, "options": [{"code", "label", "meaning"}, ...]}`.
4. One mask token is appended; one bidirectional forward pass; softmax of the diffusion head's logits at the
   mask over the option codes.

The trainers additionally re-render half of the rows (`--canonical-fraction 0.5`) with the options shuffled and
re-coded, so the model cannot rely on code positions.

### Length

Rows whose prompt plus mask token exceed 8192 tokens are **silently dropped** by `train_pack.py`
(`--max-tokens 8192`), and its shuffled-option path re-renders the raw state without compaction. Pre-render
long rows with `finetune/render_long.py` (serving's own `compact_state`) and check with `finetune/validate.py`.

## Request and response (SystemOne API)

`POST /v1/systemone`:

```json
{"model": "Nemotron-Deciffusion-8B",
 "state": {"ticket": {"subject": "Charged twice", "body": "Two charges of $49 on Mar 3."}},
 "questions": {"route": {"type": "choice", "instructions": "Which support queue should handle this ticket?",
                         "criteria": {"billing": "Payments, invoices, refunds", "technical": "Bugs and errors",
                                      "account_security": "Suspicious sign-ins, 2FA", "sales": "Upgrades and quotes"}},
               "urgent": {"type": "noul", "instructions": "Does the customer report a security incident?"}}}
```

A request may carry up to 512 questions over one state; each question is scored as its own prompt, so a
multi-question request equals several one-question training rows with the same state.

| training row | request | response `answers[key]` |
|---|---|---|
| `input.state` | `state` | – |
| `input.questions.<k>` | `questions.<k>` (any key) | – |
| `reference.target` (choice) | – | `choice` = argmax key; `probabilities` per key; `confidence` = max probability |
| `reference.target` (noul) | – | `noul` = p(true) |
| `reference.target` (score) | – | `probabilities` per level `"0"`…; `score` = expected level Σ i·p(i); `legend`; `confidence` |
| `teacher_probabilities` | – | compare with `probabilities` (or `{false: 1-noul, true: noul}`) |

To turn a production log into training rows, store the request's `state` and the question as `input`, and
the later observed outcome (or human label) as `reference.target`, keeping the outcome **out of** the state.
