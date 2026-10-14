# v2 release numbers: sources (tag `v2`)

Generated from the evaluation results at 2026-10-14 22:53. Paths are relative to the project directory (`--project-dir`) unless they start with `release/`. Method constants written in the prose (bin width, compaction budget) are not results and are not listed.

| placeholder | value | source | how |
|---|---|---|---|
| `AA_BEST_HLE` | 23.6 (Claude Haiku 5.5 (Low)) | `evals/aa_di_v1/aa_source/aa_di_v1_leaderboard.json` | max over the four reference models |
| `AA_SNAPSHOT_DATE` | 2026-10-13 | `evals/aa_di_v1/aa_source/aa_di_v1_leaderboard.json` | file modification date of the leaderboard snapshot |
| `ATTRIBUTION_LIST` | (table/list, 22 lines) | `release/v2/release.json` | attribution[] (from the data card attribution notice) |
| `AUTHOR` | Alex Steiner | `release/v2/release.json` | author |
| `BASE_MODEL` | nvidia/Nemotron-Labs-Diffusion-8B | `release/v2/release.json` | base_model |
| `BASE_MODEL_REVISION` | 16c67f0560b912e93e0cabb6e0c4f5c3086d95fc | `release/v2/release.json` | base_model_revision |
| `BLOG_URL` | https://pst2154.github.io/nemotron-deciffusion/ | `release/v2/release.json` | blog_url |
| `CKPT_SHA256` | 722396e8b46e34ba27db49584fea3b2be6c1797f62be58029c6ff142db072bc2 | `release/v2/SHA256SUMS-v2` | sha256 of model.safetensors |
| `CODE_LICENSE` | Apache License 2.0 | `release/v2/release.json` | code_license |
| `CODE_URL` | https://github.com/pst2154/nemotron-deciffusion | `release/v2/release.json` | code_url |
| `CONTAINER` | ghcr.io/pst2154/nemotron-deciffusion:8b-v2 | `release/v2/release.json` | container_image:container_tag |
| `FILLED_AT` | 2026-10-14 22:53 | `clock` | time of the fill run |
| `HF_REPO` | nemotron-deciffusion/Nemotron-Deciffusion-8B-v2 | `release/v2/release.json` | hf_repo |
| `HF_REPO_NVFP4` | nemotron-deciffusion/Nemotron-Deciffusion-8B-v2-NVFP4 | `release/v2/release.json` | hf_repo_nvfp4 |
| `HF_REVISION` | 1dea38015e86e101eb978a33fa493818fc71e366 | `release/v2/release.json` | hf_revision |
| `LAT_HL_GPU` | GB200 | `serving measurements (latency_v2_public.json)` | row with "headline": true, field for LAT_HL_GPU |
| `LAT_HL_P50` | 24.1 | `serving measurements (latency_v2_public.json)` | row with "headline": true, field for LAT_HL_P50 |
| `LAT_HL_PRECISION` | NVFP4 | `serving measurements (latency_v2_public.json)` | row with "headline": true, field for LAT_HL_PRECISION |
| `LAT_HL_TOKENS` | 362 | `serving measurements (latency_v2_public.json)` | row with "headline": true, field for LAT_HL_TOKENS |
| `LAT_MAX_DPS` | 201.1 | `serving measurements (latency_v2_public.json)` | row with the highest decisions_per_s |
| `LAT_MAX_DPS_CONC` | 16 | `serving measurements (latency_v2_public.json)` | row with the highest decisions_per_s |
| `LAT_MAX_DPS_GPU` | GB200 | `serving measurements (latency_v2_public.json)` | row with the highest decisions_per_s |
| `LAT_MAX_DPS_PRECISION` | NVFP4 | `serving measurements (latency_v2_public.json)` | row with the highest decisions_per_s |
| `LAT_MAX_DPS_TOKENS` | 362 | `serving measurements (latency_v2_public.json)` | row with the highest decisions_per_s |
| `MODEL_LICENSE_NAME` | NVIDIA Nemotron Open Model License | `release/v2/release.json` | model_license_name |
| `MODEL_LICENSE_URL` | https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/ | `release/v2/release.json` | model_license_url |
| `NVFP4_STATUS` | [nemotron-deciffusion/Nemotron-Deciffusion-8B-v2-NVFP4](https://huggingface.co/nemotron-deciffusion/Nemotron-Deciffusion-8B-v2-NVFP4), mixed NVFP4/FP8 at 6.6 effective bits for Blackwell data-center GPUs (validated on GB200), faster than FP8 there | `release/v2/release.json` | nvfp4_status |
| `N_AA_HLE` | 513 | `ov/aadi/v2.json` | ["components"]['hle_mc']["n"] |
| `N_AA_W2C` | 3,423 | `ov/aadi/v2.json` | ["components"]['when2call']["n"] |
| `N_AA_WGT` | 880 | `ov/aadi/v2.json` | ["components"]['wildguardtest']["n"] |
| `N_JEVBENCH` | 231 | `logs/f-v2.eval.done (or logs/neval-f-v2.log)` | compare.metrics()['jevbench_n'] |
| `N_V2_BABILONG_16K` | 250 | `results/long-v2.summary.json` | ["babilong-16k"]["n"] |
| `N_V2_C_BABILONG_32K` | 250 | `results/longc-v2.summary.json` | ["babilong-32k"]["n"] |
| `N_V2_QUALITY` | 300 | `results/long-v2.summary.json` | ["quality"]["n"] |
| `RELEASE_DATE` | 2026-10-14 | `release/v2/release.json` | release_date |
| `TABLE_AADI` | (table/list, 7 lines) | `ov/aadi/v2.json; evals/aa_di_v1/aa_source/aa_di_v1_leaderboard.json` | metrics block of ov/aadi/<tag>.json; AA rows: scores + calibrationBins of the leaderboard JSON |
| `TABLE_LATENCY` | (table/list, 18 lines) | `serving measurements (latency_v2_public.json)` | rows[] with fields gpu, precision, prompt_tokens, concurrency, p50_ms, p95_ms, decisions_per_s, source |
| `TABLE_LATENCY_SITE` | (table/list, 18 lines) | `serving measurements (latency_v2_public.json)` | rows[] without the measurement column (launch site) |
| `TABLE_RESULTS` | (table/list, 29 lines) | `--di-scores (Decision Index kit scores.json, full public 0.3 suite); guard_rerank/results/student-base-v2-safety.score.txt; logs/f-v2.eval.done (or logs/neval-f-v2.log); logs/judge-v2.out; logs/route-ht-v2.report; results/extra-v2.txt; results/long-v2.summary.json; results/longc-v2.summary.json; results/m-f-v2-audit/predictions.jsonl minus lineage/eval_decontam.txt; results/m-f-v2-eval/predictions.jsonl minus lineage/eval_decontam.txt; results/m-f-v2-jevbench/predictions.jsonl; results/robust-v2/inject.jsonl` | compare.metrics() for each tag; rows with no result for any tag are dropped |
| `TABLE_RESULTS_SHORT` | (table/list, 12 lines) | `--di-scores (Decision Index kit scores.json, full public 0.3 suite); guard_rerank/results/student-base-v2-safety.score.txt; logs/f-v2.eval.done (or logs/neval-f-v2.log); logs/judge-v2.out; logs/route-ht-v2.report; results/extra-v2.txt; results/long-v2.summary.json; results/longc-v2.summary.json; results/m-f-v2-audit/predictions.jsonl minus lineage/eval_decontam.txt; results/m-f-v2-eval/predictions.jsonl minus lineage/eval_decontam.txt; results/m-f-v2-jevbench/predictions.jsonl; results/robust-v2/inject.jsonl` | compare.metrics() for each tag; rows with no result for any tag are dropped |
| `TAG` | v2 | `command line` | --tag |
| `V2_AA_ECE_WGT` | 0.017 | `ov/aadi/v2.json` | ["metrics"]['aa_ece_wildguardtest'] |
| `V2_AA_HLE` | 11.7 | `ov/aadi/v2.json` | ["metrics"]['aa_hle_mc'] |
| `V2_AA_W2C` | 94.2 | `ov/aadi/v2.json` | ["metrics"]['aa_when2call'] |
| `V2_AA_WGT` | 91.1 | `ov/aadi/v2.json` | ["metrics"]['aa_wildguard'] |
| `V2_AUC_ATLAS_VERIFY` | 0.631 | `logs/route-ht-v2.report` | compare.metrics('v2')['auc_atlas_verify'] |
| `V2_AUC_AGENT_ESCALATE` | 0.575 | `logs/route-ht-v2.report` | compare.metrics('v2')['auc_agent_escalate'] |
| `V2_AUC_AGENT_VERIFY` | 0.725 | `logs/route-ht-v2.report` | compare.metrics('v2')['auc_agent_verify'] |
| `V2_AUDIT` | 618/678 | `results/m-f-v2-audit/predictions.jsonl minus lineage/eval_decontam.txt` | compare.metrics('v2')['audit'] |
| `V2_BABILONG_16K` | 98.4 | `results/long-v2.summary.json` | compare.metrics('v2')['babilong-16k'] |
| `V2_BABILONG_4K` | 99.2 | `results/long-v2.summary.json` | compare.metrics('v2')['babilong-4k'] |
| `V2_BABILONG_8K` | 98.8 | `results/long-v2.summary.json` | compare.metrics('v2')['babilong-8k'] |
| `V2_C_BABILONG_16K` | 99.2 | `results/longc-v2.summary.json` | compare.metrics('v2')['c_babilong-16k'] |
| `V2_C_BABILONG_32K` | 99.6 | `results/longc-v2.summary.json` | compare.metrics('v2')['c_babilong-32k'] |
| `V2_C_QUALITY` | 80.3 | `results/longc-v2.summary.json` | compare.metrics('v2')['c_quality'] |
| `V2_DI_ARTS` | 37.8 | `--di-scores (Decision Index kit scores.json, full public 0.3 suite)` | compare.metrics('v2')['di_arts'] |
| `V2_DI_INDEX` | 53.9 | `--di-scores (Decision Index kit scores.json, full public 0.3 suite)` | compare.metrics('v2')['di_index'] |
| `V2_DI_KNOWLEDGE` | 35.3 | `--di-scores (Decision Index kit scores.json, full public 0.3 suite)` | compare.metrics('v2')['di_knowledge'] |
| `V2_DI_LANGUAGE` | 56.9 | `--di-scores (Decision Index kit scores.json, full public 0.3 suite)` | compare.metrics('v2')['di_language'] |
| `V2_DI_RETRIEVAL` | 58.0 | `--di-scores (Decision Index kit scores.json, full public 0.3 suite)` | compare.metrics('v2')['di_retrieval'] |
| `V2_DI_TOOLS` | 80.2 | `--di-scores (Decision Index kit scores.json, full public 0.3 suite)` | compare.metrics('v2')['di_tools'] |
| `V2_EVAL` | 528/578 | `results/m-f-v2-eval/predictions.jsonl minus lineage/eval_decontam.txt` | compare.metrics('v2')['eval'] |
| `V2_INJECT` | 4.93 | `results/robust-v2/inject.jsonl` | compare.metrics('v2')['inject'] |
| `V2_JEVBENCH` | 210/231 | `logs/f-v2.eval.done (or logs/neval-f-v2.log)` | compare.metrics('v2')['jevbench'] |
| `V2_JUDGE_TOOL` | 0.981 | `logs/judge-v2.out` | compare.metrics('v2')['judge_tool'] |
| `V2_JUDGE_TRAJ` | 0.947 | `logs/judge-v2.out` | compare.metrics('v2')['judge_traj'] |
| `V2_NDCG10` | 53.1 | `results/extra-v2.txt` | compare.metrics('v2')['ndcg10'] |
| `V2_QUALITY` | 80.3 | `results/long-v2.summary.json` | compare.metrics('v2')['quality'] |
| `V2_SAF_AEGIS_PROMPT` | 87.4 | `guard_rerank/results/student-base-v2-safety.score.txt` | compare.metrics('v2')['saf_aegis_prompt'] |
| `V2_SAF_AEGIS_RESPONSE` | 83.0 | `guard_rerank/results/student-base-v2-safety.score.txt` | compare.metrics('v2')['saf_aegis_response'] |
| `V2_SAF_TOXICCHAT` | 73.5 | `guard_rerank/results/student-base-v2-safety.score.txt` | compare.metrics('v2')['saf_toxicchat'] |
| `V2_TIER_HARD` | 90/111 | `results/m-f-v2-jevbench/predictions.jsonl` | compare.metrics('v2')['tier:hard'] |
| `VLLM_PRS` | vllm-project/vllm#60283 | `release/v2/release.json` | vllm_prs |

## Values inside tables and lists

Each cell of the generated tables, with its own source (`V2_` = this release; `N_` = item counts). `<deciffusion-report>` is the DI report directory configured as `R` in `ov/compare.py`.

| value key | value | source | how |
|---|---|---|---|
| `AA_BEST_W2C` | 83.7 (GLiDE) | `evals/aa_di_v1/aa_source/aa_di_v1_leaderboard.json` | max over the four reference models |
| `AA_BEST_WGT` | 95.1 (GLiDE) | `evals/aa_di_v1/aa_source/aa_di_v1_leaderboard.json` | max over the four reference models |
| `CONTAINER_IMAGE` | ghcr.io/pst2154/nemotron-deciffusion | `release/v2/release.json` | container_image |
| `CONTAINER_TAG` | 8b-v2 | `release/v2/release.json` | container_tag |
| `LATENCY_HEADLINE` | 24.1 ms p50 on one GB200 (NVFP4, ~362-token prompts) | `serving measurements (latency_v2_public.json)` | row with "headline": true |
| `LAT_HL_DPS` | 29.0 | `serving measurements (latency_v2_public.json)` | row with "headline": true, field for LAT_HL_DPS |
| `LAT_HL_P95` | 28.7 | `serving measurements (latency_v2_public.json)` | row with "headline": true, field for LAT_HL_P95 |
| `LAT_MIN_P50` | 24.1 | `serving measurements (latency_v2_public.json)` | concurrency-1 row with the lowest p50_ms |
| `LAT_MIN_P50_GPU` | GB200 | `serving measurements (latency_v2_public.json)` | concurrency-1 row with the lowest p50_ms |
| `LAT_MIN_P50_PRECISION` | NVFP4 | `serving measurements (latency_v2_public.json)` | concurrency-1 row with the lowest p50_ms |
| `LAT_MIN_P50_TOKENS` | 362 | `serving measurements (latency_v2_public.json)` | concurrency-1 row with the lowest p50_ms |
| `N_AUDIT` | 678 | `results/m-f-v2-audit/predictions.jsonl minus lineage/eval_decontam.txt` | compare.metrics()['audit_n'] |
| `N_EVAL` | 578 | `results/m-f-v2-eval/predictions.jsonl minus lineage/eval_decontam.txt` | compare.metrics()['eval_n'] |
| `N_TIER_HARD` | 111 | `results/m-f-v2-jevbench/predictions.jsonl` | compare.metrics()['tier:hard_n'] |
| `N_V2_BABILONG_32K` | 250 | `results/long-v2.summary.json` | ["babilong-32k"]["n"] |
| `N_V2_BABILONG_4K` | 250 | `results/long-v2.summary.json` | ["babilong-4k"]["n"] |
| `N_V2_BABILONG_8K` | 250 | `results/long-v2.summary.json` | ["babilong-8k"]["n"] |
| `N_V2_C_BABILONG_16K` | 250 | `results/longc-v2.summary.json` | ["babilong-16k"]["n"] |
| `N_V2_C_BABILONG_4K` | 250 | `results/longc-v2.summary.json` | ["babilong-4k"]["n"] |
| `N_V2_C_BABILONG_8K` | 250 | `results/longc-v2.summary.json` | ["babilong-8k"]["n"] |
| `N_V2_C_QUALITY` | 300 | `results/longc-v2.summary.json` | ["quality"]["n"] |
| `V2_AA_ECE_HLE` | 0.314 | `ov/aadi/v2.json` | ["metrics"]['aa_ece_hle_mc'] |
| `V2_AA_ECE_W2C` | 0.030 | `ov/aadi/v2.json` | ["metrics"]['aa_ece_when2call'] |
| `V2_AA_OPEN` | 65.7 | `ov/aadi/v2.json` | ["metrics"]['aa_di_open'] |
