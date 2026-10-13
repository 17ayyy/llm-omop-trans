# llm-omop-trans

[简体中文](README.zh-CN.md)

## Introduction

`llm-omop-trans` generates **review-only Simplified Chinese candidate labels** for identified OMOP concepts through an OpenAI-compatible Chat Completions endpoint. It selects one `vocabulary_id` from a seven-column input, validates the complete input, and enriches eligible records from three files in one Athena snapshot. It never writes candidates into the authoritative `concept_name_zh` field.

Use it only for a licensed, explicitly authorized terminology review workflow. The bundled `DEMO` records are invented and suitable for a small connectivity test. Read the [source and transfer policy](docs/SOURCE_POLICY.md) before using real data.

## Features

- Strict UTF-8, tilde-delimited input validation; exact vocabulary selection; same-snapshot identity, validity, and existing-label checks before transfer.
- One concept per request, a fixed JSON schema, strict parsing, and deterministic checks of 14 medical attribute families. All candidates require expert review.
- Separate Pilot runs of 1–20 eligible records, a machine gate, explicit Pilot selection, and reuse of verified Pilot successes in a full run.
- Pilot results in `outputs/test/<run-id>/`, full results in `outputs/<run-id>/`, and a rebuildable `outputs/run-index.json` for run lookup.
- DuckDB call ledger, per-call commits, verified CSV export, fingerprinted resume, and no repeat of committed successes. Fallback models are used only after identified quota exhaustion and a synthetic probe.
- A report that merges compatible full runs and distinguishes API success, candidate generation, validation, and completion.

## Requirements

- Python 3.12 (`>=3.12,<3.13`) and [uv](https://docs.astral.sh/uv/).
- An OpenAI-compatible Chat Completions endpoint and a local API key for live requests.
- For real concepts, a seven-column input plus `CONCEPT.csv`, `CONCEPT_SYNONYM.csv`, and `VOCABULARY.csv` from one licensed Athena snapshot. Real data are not distributed here.

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`. The original course repository is not needed.

## Installation

```powershell
git clone https://github.com/17ayyy/llm-omop-trans.git
cd llm-omop-trans
uv sync --locked
Copy-Item .env.example .env
```

Edit `.env` with an authorized endpoint, model, and key. It initially points to the bundled synthetic tables. `.env`, real data, run state, logs, and caches are ignored by Git. `uv.toml` selects PyPI and a project-local cache; an inherited `UV_DEFAULT_INDEX` environment variable may override that index.

## Configuration

| `.env` key | Meaning |
| --- | --- |
| `OMOP_CONCEPT_PATH`, `OMOP_CONCEPT_SYNONYM_PATH`, `OMOP_VOCABULARY_PATH` | UTF-8, tab-delimited snapshot tables. Relative paths resolve from the working directory. |
| `OMOP_ATHENA_RELEASE`, `OMOP_EFFECTIVE_DATE` | Snapshot label and effective date (`YYYY-MM-DD`) for provenance and validity checks. |
| `LLM_BASE_URL` | Plain HTTP(S) OpenAI-compatible base URL, normally ending in `/v1`; no credentials, query, or fragment. |
| `LLM_MODEL`, `LLM_API_KEY` | Default model and secret. `LLM_MODEL_API_KEY` is a legacy fallback alias; `LLM_API_KEY` wins. |
| `LLM_TIMEOUT_SECONDS` | Per-request timeout, default `60`, valid range `(0, 3600]`. |
| `LLM_TRANS_THINKING_MODE` | `auto` by default: explicitly disable only for documented first-party endpoint/model pairs. `disabled` fails before a request if unsupported; `provider_default` sends no reasoning field. `enabled`, `adaptive`, and `low` require model support. |

| Provider | Models with explicit disable in `auto` | Request field | Limitation and official source |
| --- | --- | --- | --- |
| DeepSeek | `deepseek-flash`, `deepseek-v4-pro` | `thinking.type=disabled` | [Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion) |
| Qwen | Listed hybrid Qwen 3.5–3.8 models, e.g. `qwen3.7-plus` | `enable_thinking=false` | Pure thinking models cannot turn it off; [thinking mode](https://platform.qianwenai.com/docs/developer-guides/text-generation/thinking) |
| MiniMax | `MiniMax-M3` | `thinking.type=disabled` | M2.x ignores it; M3.1 Flash Preview rejects it and supports `low` instead; [Chat Completions](https://platform.minimax.cn/docs/api-reference/text-chat-openai) |
| StepFun | None | None | Supported models offer `reasoning_effort=low`, not off; [Chat Completions](https://platform.stepfun.ai/docs/en/api-reference/chat/chat-completion-create) |
| Agnes | None | None | `agnes-2.5-flash` documents `enabled`, but disabling is unverified; [Flash](https://agnes-ai.cn/zh-Hans/docs/agnes-25-flash) |
| Gemini | `gemini-2.5-flash`, `gemini-2.5-flash-lite` | `reasoning_effort=none` | Pro and Gemini 3 cannot turn it off; supported models may use `low`; [OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai) |
| OpenAI | Listed `gpt-5.1`, `gpt-5.5`, `gpt-5.6-*`, `gpt-6-sol/luna` | `reasoning_effort=none` | `gpt-6-astra` and `gpt-6.1-sol` support `low` but not `none`; [reasoning](https://developers.openai.com/api/docs/guides/reasoning) |

Only the model IDs and official endpoint paths listed in code inherit these semantics. OpenAI-compatible proxies do not automatically qualify. `adaptive` sends `thinking.type=adaptive` and `reasoning_split=true` only for MiniMax-M3. Unsupported defaults emit a warning and keep the provider default; that is not proof of disabled reasoning. New successful rows fingerprint the actual reasoning fields; historical successful rows keep their legacy fingerprints and are not retranslated. Local Mock tests verify request construction; real provider API behavior, token savings, and latency remain unverified.

`uv run llm-omop-trans translate --help` lists every flag. Key options:

| Flag | Behavior |
| --- | --- |
| `--input`, `--vocabulary-id` | Required input path and exact, case-sensitive target vocabulary. |
| `--concurrency`, `--batch-size` | Positive concurrency (default `1`) and scheduling batch size (default `20`); one concept per request. |
| `--pilot-only`, `--pilot-size`, `--pilot-label` | Separate Pilot; size defaults to `3` and must be `1..20`. A label distinguishes retries. |
| `--require-pilot`, `--approved-pilot-run-id` | Require a machine-valid Pilot and select a reviewed Pilot when needed. |
| `--resume-run-id`, `--model-name`, `--fallback-model` | Resume, override the model, or append fallback models in order (repeat the last flag). |
| `--allow-code-hash-drift`, `--allow-endpoint-switch`, `--retry-auth-failures` | Explicit audited recovery controls; endpoint switching requires a valid Pilot on the new endpoint. |

Input, Athena, prompt, validator, endpoint, or code changes may block ordinary resume. A model switch preserves prior successes and their recorded actual model names.

## Quick Start

With `.env` configured, send **one invented record** to the authorized model:

```powershell
uv run --locked llm-omop-trans translate --input .\examples\synthetic\input.csv --vocabulary-id DEMO --pilot-only --pilot-size 1 --concurrency 1 --batch-size 1
```

The command prints `outputs/test/<run-id>/omop-trans.csv`. Inspect the CSV and review files; a machine-valid Pilot is not expert approval.

## Examples

```powershell
# Pilot on an authorized input; inspect its output and run ID.
uv run --locked llm-omop-trans translate --input .\rawdata\authorized-input.csv --vocabulary-id AUTHORIZED_ID --pilot-only --pilot-size 3 --concurrency 1 --batch-size 3

# Full run only after scope, endpoint, source license, and Pilot review.
uv run --locked llm-omop-trans translate --input .\rawdata\authorized-input.csv --vocabulary-id AUTHORIZED_ID --require-pilot --approved-pilot-run-id run-PASTE_REAL_PILOT_ID --concurrency 1 --batch-size 10

# Resume without re-sending committed successes.
uv run --locked llm-omop-trans translate --input .\rawdata\authorized-input.csv --vocabulary-id AUTHORIZED_ID --resume-run-id run-PASTE_REAL_RUN_ID --require-pilot --concurrency 1 --batch-size 10

# Switch model for pending requests; a fallback is probed on quota exhaustion.
uv run --locked llm-omop-trans translate --input .\rawdata\authorized-input.csv --vocabulary-id AUTHORIZED_ID --resume-run-id run-PASTE_REAL_RUN_ID --model-name NEW_MODEL --fallback-model BACKUP_MODEL --require-pilot

# Merge compatible full runs and write a Markdown report.
uv run --locked llm-omop-trans report --input .\rawdata\authorized-input.csv
```

Replace placeholders with authorized paths, vocabulary IDs, models, and run IDs. The synthetic one-record Pilot and its resume were verified; the real-data commands are templates, not evidence of real-data authorization.

## Input & Output

Input is UTF-8 CSV with `~` separator and this **exact seven-column order**:

```text
concept_id~concept_name_zh~concept_name~domain_id~vocabulary_id~concept_class_id~model
900000001~~Left kidney~Condition~DEMO~Clinical Finding~
```

`concept_id` is a unique nonnegative integer. The complete input is checked before vocabulary selection. The original fields are retained for selected rows. Existing Chinese names, missing or inactive Athena records, and conflicting vocabulary identities are skipped before network calls.

Full runs write `outputs/<run-id>/omop-trans.csv`; Pilot runs write `outputs/test/<run-id>/omop-trans.csv`. Both preserve selected-input order and the same nine columns:

```text
concept_id~concept_name_zh~concept_name~domain_id~vocabulary_id~concept_class_id~candidate_name_zh~review_status~model_name
```

`candidate_name_zh` is a proposal; `concept_name_zh` remains the original field. A nonempty candidate passing or requiring review gets `pending_expert`; failed checks remain as evidence and are not accepted. `model_name` records the actual response model when supplied, otherwise the requested model. Each run also stores `run_manifest.json`, `state.duckdb`, and review/exception CSVs. `report` writes `outputs/omop-trans-all.csv` and `outputs/llm-trans-report.md` from compatible full runs, excluding Pilot-only runs.

`outputs/run-index.json` maps each run ID to its type, vocabulary, purpose, relative output path, status, and recorded models. `model_name` is set only when one actual model is known; `models_used` preserves every model found in the call ledger or saved successful rows. The file is rebuilt under a lock and replaced atomically when a run starts, resumes, or ends. If it is missing or damaged, rebuild it from manifests and DuckDB state with:

```powershell
uv run --locked python -m llm_omop_trans.run_index
```

Older Pilot directories under `outputs/<run-id>/` remain readable and resumable. The index marks older runs without a recorded type as `unknown` instead of guessing. Outputs and the index are local artifacts ignored by Git.

## Project Structure

```text
llm-omop-trans/
├── llm_omop_trans/
│   ├── config.py       # contract, prompt, environment, transfer denials
│   ├── llm_trans.py    # CLI, preflight, calls, Pilot, state, export
│   ├── validator.py    # 14 deterministic attribute checks
│   ├── report.py       # full-run merge and quality report
│   ├── run_index.py    # lock-protected run index and Pilot paths
│   └── utils.py        # strict JSON and hashing
├── llm_trans_main.py   # compatible Python CLI entry
├── examples/synthetic/ # invented input and three tiny snapshot tables
├── tests/             # synthetic unit and local HTTP integration tests
├── docs/SOURCE_POLICY.md
├── .env.example        # safe configuration template
├── pyproject.toml
├── uv.toml
└── uv.lock
```

Runtime output layout:

```text
outputs/
├── test/<run-id>/      # Pilot manifest, DuckDB state, CSV
├── <run-id>/           # full-run manifest, DuckDB state, CSV
└── run-index.json      # rebuildable metadata index
```

## License

**No software license has been granted or added to this repository.** Public visibility does not itself authorize reuse; obtain the owner's permission before copying, modifying, or redistributing the code. Real OMOP/Athena and other terminology data are not included and remain subject to their source licenses and the [transfer policy](docs/SOURCE_POLICY.md). Generated candidates are for internal review, not automatically approved labels or licensed redistribution.
