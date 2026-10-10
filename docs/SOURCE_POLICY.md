# Source and external-transfer policy

This standalone project carries forward the original course project's `docs/SOURCE_POLICY.md` and `docs/Permit.md` decision dated 2026-10-09. That decision permitted review-only candidate generation through the **then-configured** external LLM endpoint for eligible vocabularies. It did not permit public redistribution of source terminology or generated derivatives. A different endpoint, new data source, or broader use requires a fresh authorization and license check.

## Before a real-data request

1. Confirm the exact `vocabulary_id`, source release, local files, right to external processing, and configured endpoint. The bundled `DEMO` records are invented and require no terminology license.
2. Confirm the record has no usable Chinese name/candidate and that its Athena concept identity and effective date agree with the input. Do not use English-name matching as proof of identity.
3. Start with a small Pilot (`--pilot-size` 1–20), inspect the candidate and validation evidence, and use `--require-pilot` for a later full run. Keep requests bounded to the authorized scope.
4. Keep `.env`, licensed snapshots, inputs, `outputs/`, logs, and DuckDB state local. Do not publish them. Model output remains `candidate_name_zh` and `pending_expert` until expert review.

The CLI blocks external transfer for `CPT4`, `RxNorm`, `RxNorm Extension`, `LOINC`, `OSM`, `ICD9Proc`, `SNOMED`, `ICD10CM`, and literal missing-value markers such as `nan` and `null`. The historical course scope also excluded already processed `dm+d` and `NDC` from its remaining batch; their past inclusion in a permit is not an instruction to retransmit them.

The original pipeline's source matching and `dm+d` rules are separate from this CLI. This repository contains neither its source registry nor a real Athena snapshot. For real data, the operator must provide the three same-release tables and document the specific transfer permission. The CLI's technical Pilot gate is not a legal or expert-approval gate.
