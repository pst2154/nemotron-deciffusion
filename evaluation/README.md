# Evaluation

The evaluators behind the published numbers. They use the serving prompt builder (`serving/app/scoring.py`), so
evaluation and serving score exactly the same prompts.

| file | what it does |
|---|---|
| `native_eval.py` | Single-pass evaluation (prompt + one mask token) of a checkpoint on JevBench and on JSONL decision splits. Writes `results/<tag>-<split>/predictions.jsonl`. |
| `jev16.py` | JevBench top-1 and the v1.6-style proxy score from those predictions (`jev16.py <harness> <results>/<tag>`). |
| `proxy_v16.py` | The same proxy for labelled local predictions. |
| `di_engine.py` | Engine for the Decision Index kit: runs the same single-pass read for every Decision Index request. |
| `aa_open_components.py` | AA Decision Index open components (HLE-MC, When2Call, WildGuardTest) and calibration (ECE): builds the test sets from Hugging Face, runs them through `di_engine.py` and scores them. Task ids are in `aa_open_ids.json`. |

```bash
export JEVBENCH_HARNESS=<path to the JevBench harness checkout>
export DECIFFUSION_EVAL_DATA=<directory with eval.jsonl / audit.jsonl>   # optional extra splits
python3 evaluation/native_eval.py --checkpoint <checkpoint dir> --tag v2 --splits jevbench --attention noncausal
python3 evaluation/jev16.py $JEVBENCH_HARNESS results/v2        # top-1 and v1.6-style proxy
```

Use `--attention noncausal`, as the model is trained and served with bidirectional attention. Run every evaluation in
BF16.

## AA Decision Index: open components

Artificial Analysis's Decision Index combines four benchmarks. `aa_open_components.py` runs the three public ones with
[AA's published methodology](https://artificialanalysis.ai/methodology/decision-index):

| component | items | metric |
|---|---|---|
| HLE-MC (`cais/hle`) | the 513 text-only multiple-choice questions AA lists; question text before "Answer Choices:", lettered options in their original order | accuracy (exact letter) |
| When2Call (`nvidia/When2Call`, test MCQ) | 3,423 tasks: the 3,652 test tasks minus the 229 "request for info" tasks AA filters out; options A-D = direct answer, tool call, request for info, cannot answer | accuracy |
| WildGuardTest (`allenai/wildguardmix`, `wildguardtest`) | 880 items on which all three annotators agree on all three labels; three yes/no questions each: is the request harmful, is the reply a refusal, is the reply harmful | mean F1 over the three questions (positive = yes, yes when p(yes) >= 0.5) |

- **Mean of the 3 open components** = (HLE-MC accuracy + When2Call accuracy + WildGuardTest F1) / 3, printed by `score` and
  written to `scores.json` as `mean_of_3_open_components`.
- **Calibration (ECE)** follows AA: confidence is the probability of the chosen answer (yes/no: max(p, 1 - p)), bins of 0.1,
  only bins with at least 20 answers, answer-weighted mean of |accuracy - mean confidence|.
- The fourth component, AA-Omniscience-Decision, is private, so the script reports the three open components and their mean,
  not an overall AA index.
- The datasets are downloaded from Hugging Face at pinned revisions and checked by sha256; only the task ids
  (`aa_open_ids.json`) are kept in this repo. `cais/hle` and `allenai/wildguardmix` are gated: accept their terms on
  Hugging Face and log in first. The test rows are for evaluation only.

```bash
pip install pyarrow huggingface_hub "git+https://github.com/apolinario/decision-index"   # the Decision Index kit, for di_engine.py
hf auth login                                                                           # after accepting the cais/hle and allenai/wildguardmix terms
python3 evaluation/aa_open_components.py build --data aa_data
python3 evaluation/aa_open_components.py run   --data aa_data --checkpoint nemotron-deciffusion/Nemotron-Deciffusion-8B-v2 --out results/aa-v2
python3 evaluation/aa_open_components.py score --data aa_data --out results/aa-v2   # per-component scores, mean of the 3, ECE
```

`--checkpoint` takes the Hugging Face id or a local checkpoint directory. `run` uses BF16 and bidirectional attention
(`--attention noncausal`, the default) and resumes from `results/aa-v2/<component>.jsonl`; `--only` limits `build` and `run`
to some components.
