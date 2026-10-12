"""Generate review-only OMOP Chinese candidates with resumable state."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast

import duckdb
import httpx
import openai
from filelock import FileLock, Timeout
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

from llm_omop_trans.config import (
    _SYSTEM_PROMPT,
    EXTERNAL_LLM_DENIED_VOCABULARY_IDS,
    INPUT_COLUMNS,
    INVALID_VOCABULARY_SENTINELS,
    OUTPUT_COLUMNS,
    OUTPUT_ROOT,
    PROMPT_VERSION,
    SCRIPT_VERSION,
    VALIDATOR_VERSION,
    RuntimeConfig,
    StandaloneError,
    _load_runtime,
)
from llm_omop_trans.run_index import (
    pilot_directories,
    rebuild_index,
    run_directory,
)
from llm_omop_trans.utils import (
    _canonical_json,
    _implementation_hash,
    _load_strict_json,
    _sha256_bytes,
    _sha256_file,
    _sql_string,
)
from llm_omop_trans.validator import (
    CandidateValidation,
    ValidationStatus,
    validate_candidate,
)


class StrictModel(BaseModel):
    """Reject coercion and unknown fields at every public LLM boundary."""

    model_config = ConfigDict(extra="forbid", strict=True)


class LlmConceptRequest(StrictModel):
    """Contain the fixed terminology context sent for one Concept."""

    concept_id: int = Field(ge=0)
    concept_name: str
    concept_code: str
    vocabulary_id: str
    vocabulary_version: str | None = None
    domain_id: str
    concept_class_id: str
    standard_concept: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    parents: list[str] = Field(default_factory=list)
    attributes: dict[str, JsonValue] = Field(default_factory=dict)
    authoritative_candidates: list[str] = Field(default_factory=list)
    input_fingerprint: str

    @field_validator(
        "concept_name",
        "concept_code",
        "vocabulary_id",
        "domain_id",
        "concept_class_id",
        "input_fingerprint",
    )
    @classmethod
    def require_nonempty_text(cls, value: str) -> str:
        """Require a nonblank identity without changing source spelling."""
        if not value.strip():
            raise ValueError("required terminology text must not be empty")
        return value


class LlmCandidateResponse(StrictModel):
    """Represent the only JSON object accepted from the model."""

    concept_name_zh: str | None
    alternatives: list[str]
    preserved_modifiers: list[str]
    ambiguity_flags: list[str]
    missing_information: list[str]
    needs_review: bool

    @field_validator("concept_name_zh")
    @classmethod
    def reject_blank_candidate(cls, value: str | None) -> str | None:
        """Disallow an empty string as a candidate."""
        if value is not None and not value.strip():
            raise ValueError("concept_name_zh must be null or non-empty")
        return value

    @field_validator(
        "alternatives",
        "preserved_modifiers",
        "ambiguity_flags",
        "missing_information",
    )
    @classmethod
    def reject_blank_list_items(cls, values: list[str]) -> list[str]:
        """Disallow empty evidence list entries."""
        if any(not value.strip() for value in values):
            raise ValueError("response lists must not contain blank strings")
        return values

    @model_validator(mode="after")
    def require_review_for_null_candidate(self) -> LlmCandidateResponse:
        """Require review when the model abstains."""
        if self.concept_name_zh is None and not self.needs_review:
            raise ValueError("a null candidate must set needs_review=true")
        return self


def _build_user_content(request: LlmConceptRequest) -> str:
    """Preserve the established single-Concept user prompt."""
    schema_json = _canonical_json(LlmCandidateResponse.model_json_schema())
    input_json = _canonical_json(request.model_dump(mode="json"))
    return (
        f"PROMPT_VERSION={PROMPT_VERSION}\n"
        f"OUTPUT_SCHEMA_JSON={schema_json}\n"
        f"INPUT_JSON={input_json}"
    )


@dataclass(frozen=True, slots=True)
class InputRow:
    """Retain one selected row and its original input text."""

    seq: int
    concept_id: int
    values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EnrichedRow:
    """Keep Athena context and a safe skip reason for one target."""

    input: InputRow
    concept_name: str | None
    concept_code: str | None
    vocabulary_id: str | None
    vocabulary_version: str | None
    domain_id: str | None
    concept_class_id: str | None
    standard_concept: str | None
    valid_start_date: str | None
    valid_end_date: str | None
    invalid_reason: str | None
    synonyms: tuple[str, ...]
    drift: tuple[str, ...]
    skip_reason: str | None


@dataclass(frozen=True, slots=True)
class ParsedResponse:
    """Contain a strict candidate and token usage from LangChain."""

    candidate: LlmCandidateResponse
    usage: dict[str, int | None]
    model_name: str | None = None


@dataclass(frozen=True, slots=True)
class CallOutcome:
    """Contain a persisted call outcome without raw provider content."""

    candidate: LlmCandidateResponse | None
    validation: CandidateValidation | None
    error_type: str | None
    http_status: int | None
    attempts: int
    latency_ms: float
    usage: dict[str, int | None]
    call_id: str
    model_name: str | None = None
    provider_error_code: str | None = None


def _validate_input(
    input_csv: Path,
    vocabulary_id: str,
) -> tuple[list[InputRow], int, str]:
    """Validate every record before selecting the target vocabulary.

    Args:
        input_csv: UTF-8 tilde-delimited input with the exact seven columns.
        vocabulary_id: Exact, case-sensitive vocabulary selection value.

    Returns:
        Selected rows, total input count, and whole-file SHA-256.

    Raises:
        StandaloneError: On encoding, shape, header, ID, or duplicate errors.
    """
    if not input_csv.is_file():
        raise StandaloneError("Input CSV does not exist")
    digest = _sha256_file(input_csv)
    selected: list[InputRow] = []
    seen: set[int] = set()
    total = 0
    try:
        with input_csv.open(
            "r",
            encoding="utf-8",
            errors="strict",
            newline="",
        ) as source:
            reader = csv.reader(source, delimiter="~", strict=True)
            if tuple(next(reader, ())) != INPUT_COLUMNS:
                raise StandaloneError(
                    "Input CSV header must match seven columns"
                )
            for row in reader:
                total += 1
                if len(row) != len(INPUT_COLUMNS):
                    raise StandaloneError(
                        f"Record {total}: expected 7 fields, got {len(row)}"
                    )
                if re.fullmatch(r"[0-9]+", row[0]) is None:
                    raise StandaloneError(
                        f"Input record {total} has an invalid concept_id"
                    )
                try:
                    concept_id = int(row[0])
                except ValueError as error:
                    raise StandaloneError(
                        f"Input record {total} has an invalid concept_id"
                    ) from error
                if concept_id in seen:
                    raise StandaloneError(
                        f"Input record {total} repeats a concept_id"
                    )
                seen.add(concept_id)
                if row[4] == vocabulary_id:
                    selected.append(InputRow(total, concept_id, tuple(row)))
    except (UnicodeError, csv.Error) as error:
        raise StandaloneError(
            "Input CSV is not valid UTF-8 tilde CSV"
        ) from error
    return selected, total, digest


def _verify_athena(config: RuntimeConfig) -> dict[str, str]:
    """Check snapshot headers and hash the three required Athena files."""
    required_headers = {
        "concept": (
            config.concept_path,
            {
                "concept_id",
                "concept_name",
                "domain_id",
                "vocabulary_id",
                "concept_class_id",
                "standard_concept",
                "concept_code",
                "valid_start_date",
                "valid_end_date",
                "invalid_reason",
            },
        ),
        "synonym": (
            config.synonym_path,
            {"concept_id", "concept_synonym_name", "language_concept_id"},
        ),
        "vocabulary": (
            config.vocabulary_path,
            {"vocabulary_id", "vocabulary_version"},
        ),
    }
    hashes: dict[str, str] = {}
    for label, (path, required) in required_headers.items():
        if not path.is_file():
            raise StandaloneError(f"Athena {label} file is missing")
        try:
            with path.open("r", encoding="utf-8", newline="") as source:
                header = set(source.readline().rstrip("\r\n").split("\t"))
        except UnicodeError as error:
            raise StandaloneError(
                f"Athena {label} header is not UTF-8"
            ) from error
        if not required <= header:
            raise StandaloneError(
                f"Athena {label} header lacks required columns"
            )
        hashes[label] = _sha256_file(path)
    return hashes


def _athena_scan(path: Path) -> str:
    """Scan an Athena TAB file without interpreting embedded quotes."""
    return (
        f"read_csv({_sql_string(path.as_posix())}, "
        f"delim={_sql_string(chr(9))}, "
        "header=true, all_varchar=true, nullstr='', "
        "strict_mode=true, quote='', escape='')"
    )


def _athena_date(value: str | None) -> date | None:
    """Parse an Athena YYYYMMDD or ISO date, if valid."""
    if not value:
        return None
    try:
        if re.fullmatch(r"[0-9]{8}", value):
            return date(int(value[:4]), int(value[4:6]), int(value[6:]))
        return date.fromisoformat(value)
    except ValueError:
        return None


def _enrich_targets(
    selected: Sequence[InputRow],
    config: RuntimeConfig,
) -> list[EnrichedRow]:
    """Join selected IDs to the exact Athena snapshot and sorted synonyms."""
    if not selected:
        return []
    connection = duckdb.connect()
    try:
        connection.execute(
            "CREATE TEMP TABLE target_ids (seq BIGINT, concept_id VARCHAR)"
        )
        connection.executemany(
            "INSERT INTO target_ids VALUES (?, ?)",
            [(row.seq, str(row.concept_id)) for row in selected],
        )
        concept_query = (
            "SELECT t.seq, c.concept_id, c.concept_name, c.concept_code, "
            "c.vocabulary_id, c.domain_id, c.concept_class_id, "
            "c.standard_concept, c.valid_start_date, c.valid_end_date, "
            "c.invalid_reason FROM "
            + _athena_scan(config.concept_path)
            + " c JOIN target_ids t ON c.concept_id = t.concept_id"
        )
        hits: dict[int, tuple[object, ...]] = {}
        for result in connection.execute(concept_query).fetchall():
            seq = int(result[0])
            if seq in hits:
                raise StandaloneError("Athena CONCEPT has duplicate target IDs")
            hits[seq] = result
        vocabulary_query = (
            "SELECT vocabulary_id, vocabulary_version FROM "
            + _athena_scan(config.vocabulary_path)
        )
        versions: dict[str, str | None] = {}
        for value, version in connection.execute(vocabulary_query).fetchall():
            if value in versions:
                raise StandaloneError("Athena VOCABULARY has duplicate IDs")
            versions[str(value)] = str(version) if version else None
        synonym_query = (
            "SELECT concept_id, concept_synonym_name FROM ("
            "SELECT s.concept_id, s.concept_synonym_name, "
            "row_number() OVER (PARTITION BY s.concept_id "
            "ORDER BY s.concept_synonym_name) AS rank "
            "FROM "
            + _athena_scan(config.synonym_path)
            + " s JOIN target_ids t ON s.concept_id = t.concept_id "
            "WHERE nullif(trim(s.concept_synonym_name), '') IS NOT NULL"
            ") WHERE rank <= 20 ORDER BY concept_id, concept_synonym_name"
        )
        synonyms: dict[str, list[str]] = {}
        for concept_id, synonym in connection.execute(synonym_query).fetchall():
            synonyms.setdefault(str(concept_id), []).append(str(synonym))
    finally:
        connection.close()

    enriched: list[EnrichedRow] = []
    for row in selected:
        match = hits.get(row.seq)
        if match is None:
            enriched.append(
                EnrichedRow(
                    row,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    (),
                    (),
                    "athena_not_found",
                )
            )
            continue
        (
            _,
            _,
            name,
            code,
            vocabulary,
            domain,
            concept_class,
            standard,
            start,
            end,
            invalid,
        ) = match
        normalized = [
            str(value) if value is not None else None
            for value in (
                name,
                code,
                vocabulary,
                domain,
                concept_class,
                standard,
                start,
                end,
                invalid,
            )
        ]
        (
            official_name,
            concept_code,
            athena_vocabulary,
            athena_domain,
            athena_class,
            standard_concept,
            start_text,
            end_text,
            invalid_reason,
        ) = normalized
        drift = []
        for label, original, official in (
            ("concept_name", row.values[2], official_name),
            ("domain_id", row.values[3], athena_domain),
            ("vocabulary_id", row.values[4], athena_vocabulary),
            ("concept_class_id", row.values[5], athena_class),
        ):
            if original != official:
                drift.append(label)
        skip: str | None = None
        if row.values[1].strip():
            skip = "existing_chinese_name"
        elif athena_vocabulary != row.values[4]:
            skip = "vocabulary_conflict"
        elif not official_name or not official_name.strip():
            skip = "official_name_unavailable"
        elif invalid_reason:
            skip = "concept_invalid"
        elif (
            _athena_date(start_text) is None
            or _athena_date(end_text) is None
            or not (
                cast(date, _athena_date(start_text))
                <= config.effective_date
                <= cast(date, _athena_date(end_text))
            )
        ):
            skip = "concept_inactive_or_invalid_date"
        elif not all((concept_code, athena_domain, athena_class)):
            skip = "athena_identity_unavailable"
        enriched.append(
            EnrichedRow(
                row,
                official_name,
                concept_code,
                athena_vocabulary,
                versions.get(athena_vocabulary or ""),
                athena_domain,
                athena_class,
                standard_concept,
                start_text,
                end_text,
                invalid_reason,
                tuple(synonyms.get(str(row.concept_id), [])[:20]),
                tuple(drift),
                skip,
            )
        )
    return enriched


def _build_request(row: EnrichedRow) -> LlmConceptRequest:
    """Construct the original request schema from Athena context."""
    if row.skip_reason is not None:
        raise StandaloneError("Cannot request a skipped target")
    fingerprint = _sha256_bytes(
        _canonical_json(
            {
                "input": row.input.values,
                "athena": (
                    row.concept_name,
                    row.concept_code,
                    row.vocabulary_id,
                    row.domain_id,
                    row.concept_class_id,
                    row.standard_concept,
                    row.valid_start_date,
                    row.valid_end_date,
                    row.synonyms,
                ),
            }
        ).encode("utf-8")
    )
    return LlmConceptRequest(
        concept_id=row.input.concept_id,
        concept_name=cast(str, row.concept_name),
        concept_code=cast(str, row.concept_code),
        vocabulary_id=cast(str, row.vocabulary_id),
        vocabulary_version=row.vocabulary_version,
        domain_id=cast(str, row.domain_id),
        concept_class_id=cast(str, row.concept_class_id),
        standard_concept=row.standard_concept,
        synonyms=list(row.synonyms),
        attributes={
            "valid_start_date": row.valid_start_date,
            "valid_end_date": row.valid_end_date,
        },
        input_fingerprint=fingerprint,
    )


def _request_fingerprint(
    request: LlmConceptRequest,
    config: RuntimeConfig,
) -> str:
    """Bind a request to model, endpoint, prompt, schema, and validator."""
    payload = {
        "request": request.model_dump(mode="json"),
        "model": config.model,
        "endpoint": config.base_url,
        "thinking_mode": config.thinking_mode,
        "prompt_version": PROMPT_VERSION,
        "system_prompt": _SYSTEM_PROMPT,
        "response_schema": LlmCandidateResponse.model_json_schema(),
        "validator_version": VALIDATOR_VERSION,
    }
    if config.model == "MiniMax-M3" and config.thinking_mode == "adaptive":
        payload["reasoning_split"] = True
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


class _ResponseError(ValueError):
    """Identify a strict parse failure without retaining provider text."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


def _parse_response(message: object) -> ParsedResponse:
    """Strictly parse one completed LangChain chat response."""
    metadata = getattr(message, "response_metadata", None)
    if not isinstance(metadata, dict):
        raise _ResponseError("invalid_response_schema")
    finish = metadata.get("finish_reason")
    if finish is not None and finish != "stop":
        raise _ResponseError("incomplete_response")
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise _ResponseError("invalid_response_schema")
    try:
        payload = _load_strict_json(content)
    except (ValueError, json.JSONDecodeError) as error:
        raise _ResponseError("invalid_candidate_json") from error
    if not isinstance(payload, dict):
        raise _ResponseError("invalid_candidate_schema")
    try:
        candidate = LlmCandidateResponse.model_validate(payload)
    except ValidationError as error:
        raise _ResponseError("invalid_candidate_schema") from error
    raw_usage = getattr(message, "usage_metadata", None) or {}
    if not isinstance(raw_usage, dict):
        raise _ResponseError("invalid_response_schema")
    usage: dict[str, int | None] = {}
    for output, source in (
        ("prompt_tokens", "input_tokens"),
        ("completion_tokens", "output_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        value = raw_usage.get(source)
        if value is not None and (type(value) is not int or value < 0):
            raise _ResponseError("invalid_response_schema")
        usage[output] = value
    reported_model = metadata.get("model_name") or metadata.get("model")
    if reported_model is not None and not isinstance(reported_model, str):
        raise _ResponseError("invalid_response_schema")
    return ParsedResponse(candidate, usage, reported_model)


def _build_chain(
    config: RuntimeConfig,
) -> Runnable[dict[str, str], ParsedResponse]:
    """Build the LangChain prompt, Chat Completions model, and strict parser."""
    extra_body = (
        {"thinking": {"type": config.thinking_mode}}
        if config.thinking_mode
        else None
    )
    if config.model == "MiniMax-M3" and config.thinking_mode == "adaptive":
        extra_body = {**extra_body, "reasoning_split": True}
    model = ChatOpenAI(
        model=config.model,
        api_key=SecretStr(config.api_key),
        base_url=config.base_url,
        temperature=0,
        timeout=config.timeout_seconds,
        max_retries=0,
        streaming=False,
        use_responses_api=False,
        extra_body=extra_body,
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM_PROMPT), ("human", "{user_content}")]
    )
    return cast(
        Runnable[dict[str, str], ParsedResponse],
        prompt
        | model.bind(response_format={"type": "json_object"}, stream=False)
        | RunnableLambda(_parse_response),
    )


def _classify_exception(error: Exception) -> tuple[str, int | None, bool]:
    """Map provider exceptions to safe categories and retry decisions."""
    if isinstance(error, _ResponseError):
        return (
            error.category,
            None,
            error.category
            in {"invalid_candidate_json", "invalid_candidate_schema"},
        )
    if isinstance(error, (openai.APITimeoutError, httpx.TimeoutException)):
        return "network_timeout", None, True
    if isinstance(error, (openai.APIConnectionError, httpx.NetworkError)):
        return "network_error", None, True
    if isinstance(error, openai.APIStatusError):
        status = error.status_code
        retryable = status in {408, 409, 425, 429} or 500 <= status <= 599
        try:
            detail = error.response.json()
        except (ValueError, TypeError):
            detail = {}
        provider_error = (
            detail.get("error", {}) if isinstance(detail, dict) else {}
        )
        if not isinstance(provider_error, dict):
            provider_error = {}
        code = str(provider_error.get("code", "")).casefold()
        message = str(provider_error.get("message", "")).casefold()
        if status == 402:
            return "quota_exhausted", status, False
        if status in {400, 429} and (
            code
            in {
                "insufficient_quota",
                "quota_exceeded",
                "insufficient_balance",
                "balance_exhausted",
                "credits_exhausted",
            }
            or any(
                phrase in message
                for phrase in (
                    "quota exhausted",
                    "quota exceeded",
                    "insufficient balance",
                    "insufficient quota",
                    "balance exhausted",
                    "credits exhausted",
                )
            )
        ):
            return "quota_exhausted", status, False
        if status == 400:
            category = "bad_request"
        elif status in {401, 403}:
            category = "authentication_error"
        elif status == 408:
            category = "request_timeout"
        elif status == 409:
            category = "conflict"
        elif status == 425:
            category = "too_early"
        elif status == 429:
            category = "rate_limit"
        elif status >= 500:
            category = "server_error"
        else:
            category = "http_error"
        return category, status, retryable
    return "transport_error", None, False


def _provider_error_code(error: Exception) -> str | None:
    """Keep a bounded provider code without retaining response text."""
    if not isinstance(error, openai.APIStatusError):
        return None
    try:
        detail = error.response.json()
    except (ValueError, TypeError):
        return None
    if not isinstance(detail, dict):
        return None
    provider_error = detail.get("error")
    if not isinstance(provider_error, dict):
        return None
    code = provider_error.get("code")
    if isinstance(code, (str, int)) and re.fullmatch(
        r"[A-Za-z0-9_.-]{1,80}", str(code)
    ):
        return str(code)
    return None


def _invoke_with_retry(
    chain: Runnable[dict[str, str], ParsedResponse],
    request: LlmConceptRequest,
) -> CallOutcome:
    """Invoke at most three times, retrying only transient failures."""
    started = time.perf_counter()
    call_id = str(uuid.uuid4())
    empty_usage: dict[str, int | None] = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }
    for attempt in range(1, 4):
        try:
            response = chain.invoke(
                {"user_content": _build_user_content(request)}
            )
        except Exception as error:
            category, status, retryable = _classify_exception(error)
            if retryable and attempt < 3:
                time.sleep(0.5 * 2 ** (attempt - 1))
                continue
            return CallOutcome(
                None,
                None,
                category,
                status,
                attempt,
                max(0.0, (time.perf_counter() - started) * 1000),
                empty_usage,
                call_id,
                provider_error_code=_provider_error_code(error),
            )
        candidate = response.candidate
        validation = (
            validate_candidate(request.concept_name, candidate.concept_name_zh)
            if candidate.concept_name_zh is not None
            else None
        )
        return CallOutcome(
            candidate,
            validation,
            None,
            200,
            attempt,
            max(0.0, (time.perf_counter() - started) * 1000),
            response.usage,
            call_id,
            response.model_name,
        )
    raise AssertionError("bounded retry loop ended unexpectedly")


def _decide_status(outcome: CallOutcome) -> tuple[str, str]:
    """Keep candidate text separate from expert approval."""
    if outcome.candidate is None:
        return "", "not_required"
    candidate = outcome.candidate.concept_name_zh
    if candidate is None:
        return "", "not_required"
    if (
        outcome.validation is None
        or outcome.validation.validation_status == ValidationStatus.FAIL
    ):
        return candidate, "not_required"
    return candidate, "pending_expert"


def _semantic_manifest(
    input_hash: str,
    vocabulary_id: str,
    hashes: Mapping[str, str],
    config: RuntimeConfig,
) -> dict[str, object]:
    """Describe stable run inputs; each call records its own model."""
    return {
        "input_sha256": input_hash,
        "vocabulary_id": vocabulary_id,
        "athena_release": config.release,
        "athena_effective_date": config.effective_date.isoformat(),
        "athena_file_sha256": dict(hashes),
        "script_version": SCRIPT_VERSION,
        "script_sha256": _implementation_hash(),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": _sha256_bytes(_SYSTEM_PROMPT.encode("utf-8")),
        "response_schema_sha256": _sha256_bytes(
            _canonical_json(LlmCandidateResponse.model_json_schema()).encode(
                "utf-8"
            )
        ),
        "validator_version": VALIDATOR_VERSION,
        "base_url": config.base_url,
    }


def _resolve_resume_semantic(
    semantic: dict[str, object],
    resume_run_id: str | None,
    allow_code_hash_drift: bool,
    allow_endpoint_switch: bool = False,
) -> dict[str, object]:
    """Reuse a checkpoint with explicit, independently gated drift."""
    if resume_run_id is None:
        if allow_code_hash_drift:
            raise StandaloneError(
                "--allow-code-hash-drift requires --resume-run-id"
            )
        if allow_endpoint_switch:
            raise StandaloneError(
                "--allow-endpoint-switch requires --resume-run-id"
            )
        return semantic
    if re.fullmatch(r"run-[0-9a-f]{16}", resume_run_id) is None:
        raise StandaloneError("Invalid resume run ID")
    path = run_directory(
        OUTPUT_ROOT,
        resume_run_id,
        str(semantic.get("run_kind")),
        resume=True,
    ) / "run_manifest.json"
    if not path.is_file():
        raise StandaloneError("Resume run is missing manifest")
    try:
        manifest = _load_strict_json(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("invalid manifest")
        previous = manifest["semantic"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise StandaloneError("Resume manifest is invalid") from error
    if not isinstance(previous, dict):
        raise StandaloneError("Resume semantic identity is invalid")
    expected = (
        "run-" + _sha256_bytes(_canonical_json(previous).encode("utf-8"))[:16]
    )
    if expected != resume_run_id:
        raise StandaloneError("Resume manifest run ID is invalid")
    if previous == semantic:
        return semantic
    prior_other = dict(previous)
    current_other = dict(semantic)
    prior_hash = prior_other.pop("script_sha256", None)
    current_hash = current_other.pop("script_sha256", None)
    prior_endpoint = prior_other.pop("base_url", None)
    current_endpoint = current_other.pop("base_url", None)
    if (
        prior_other != current_other
        or not isinstance(prior_hash, str)
        or not isinstance(current_hash, str)
        or not isinstance(prior_endpoint, str)
        or not isinstance(current_endpoint, str)
    ):
        raise StandaloneError(
            "Resume changed more than the audited script or endpoint"
        )
    if prior_hash != current_hash and not allow_code_hash_drift:
        raise StandaloneError("Resume script hash changed without approval")
    if prior_endpoint != current_endpoint and not allow_endpoint_switch:
        raise StandaloneError("Resume endpoint changed without approval")
    return previous


def _row_json(row: EnrichedRow) -> str:
    """Serialize Athena evidence and drift without losing source fields."""
    return _canonical_json(
        {
            "concept_name": row.concept_name,
            "concept_code": row.concept_code,
            "vocabulary_id": row.vocabulary_id,
            "vocabulary_version": row.vocabulary_version,
            "domain_id": row.domain_id,
            "concept_class_id": row.concept_class_id,
            "standard_concept": row.standard_concept,
            "valid_start_date": row.valid_start_date,
            "valid_end_date": row.valid_end_date,
            "invalid_reason": row.invalid_reason,
            "synonyms": row.synonyms,
            "drift": row.drift,
            "skip_reason": row.skip_reason,
        }
    )


def _create_state(
    connection: duckdb.DuckDBPyConnection,
    rows: Sequence[EnrichedRow],
    config: RuntimeConfig,
    semantic_hash: str,
) -> None:
    """Create the run tables and retain all target rows in source order."""
    connection.execute(
        "CREATE TABLE run_identity (semantic_hash VARCHAR NOT NULL)"
    )
    connection.execute("INSERT INTO run_identity VALUES (?)", [semantic_hash])
    connection.execute(
        "CREATE TABLE targets ("
        "seq BIGINT PRIMARY KEY, concept_id VARCHAR UNIQUE NOT NULL, "
        "original_name_zh VARCHAR, concept_name VARCHAR, "
        "domain_id VARCHAR, vocabulary_id VARCHAR, "
        "concept_class_id VARCHAR, input_model VARCHAR, "
        "model_name VARCHAR, enriched_json VARCHAR NOT NULL, "
        "request_fp VARCHAR, "
        "candidate_name_zh VARCHAR NOT NULL, review_status VARCHAR NOT NULL, "
        "tech_status VARCHAR NOT NULL, response_json VARCHAR, "
        "validation_json VARCHAR, metadata_json VARCHAR)"
    )
    connection.execute(
        "CREATE TABLE calls (call_id VARCHAR PRIMARY KEY, "
        "seq BIGINT NOT NULL, request_fp VARCHAR NOT NULL, "
        "model_name VARCHAR NOT NULL, outcome VARCHAR NOT NULL, "
        "error_type VARCHAR, "
        "http_status INTEGER, attempts INTEGER NOT NULL, "
        "latency_ms DOUBLE NOT NULL, usage_json VARCHAR NOT NULL, "
        "completed_at TIMESTAMP DEFAULT current_timestamp)"
    )
    data: list[tuple[object, ...]] = []
    for row in rows:
        request_fp = (
            _request_fingerprint(_build_request(row), config)
            if row.skip_reason is None
            else None
        )
        values = row.input.values
        data.append(
            (
                row.input.seq,
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                values[5],
                values[6],
                None,
                _row_json(row),
                request_fp,
                "",
                "not_required",
                row.skip_reason or "ready",
                None,
                None,
                None,
            )
        )
    if data:
        connection.executemany(
            "INSERT INTO targets VALUES ("
            "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            data,
        )


def _check_resume(
    connection: duckdb.DuckDBPyConnection,
    rows: Sequence[EnrichedRow],
    config: RuntimeConfig,
    semantic_hash: str,
    legacy_endpoint: str | None = None,
) -> None:
    """Reject stale state, mismatched rows, or incomplete success cache."""
    identity = connection.execute(
        "SELECT semantic_hash FROM run_identity"
    ).fetchall()
    if identity != [(semantic_hash,)]:
        raise StandaloneError("State semantic identity does not match")
    saved = connection.execute(
        "SELECT seq, concept_id, enriched_json, request_fp, "
        "tech_status, response_json, validation_json, model_name, "
        "metadata_json FROM targets ORDER BY seq"
    ).fetchall()
    if len(saved) != len(rows):
        raise StandaloneError("State target count does not match")
    for stored, row in zip(saved, rows, strict=True):
        (
            seq,
            concept_id,
            enriched_json,
            request_fp,
            tech,
            response,
            validation,
            model_name,
            metadata_json,
        ) = stored
        if (
            seq != row.input.seq
            or concept_id != row.input.values[0]
            or enriched_json != _row_json(row)
        ):
            raise StandaloneError("State target identity does not match")
        if tech == "succeeded":
            if not model_name or not request_fp or not metadata_json:
                raise StandaloneError("Success cache lacks model provenance")
            try:
                metadata = _load_strict_json(metadata_json)
                if not isinstance(metadata, dict):
                    raise ValueError("invalid metadata")
                requested = metadata["requested_model"]
                thinking = metadata["thinking_mode"]
                if (
                    not isinstance(requested, str)
                    or not requested
                    or thinking is not None
                    and not isinstance(thinking, str)
                    or metadata["model_name"] != model_name
                ):
                    raise ValueError("invalid model provenance")
            except (ValueError, KeyError, TypeError) as error:
                raise StandaloneError(
                    "Success cache has invalid model provenance"
                ) from error
            used_config = dataclass_replace(
                config,
                model=requested,
                thinking_mode=thinking,
                base_url=metadata.get(
                    "endpoint", legacy_endpoint or config.base_url
                ),
            )
            if request_fp != _request_fingerprint(
                _build_request(row), used_config
            ):
                raise StandaloneError(
                    "Success cache request fingerprint does not match"
                )
            if not response or not isinstance(response, str):
                raise StandaloneError("Success cache is missing a response")
            try:
                cached = _load_strict_json(response)
                candidate = LlmCandidateResponse.model_validate(cached)
            except (ValueError, ValidationError) as error:
                raise StandaloneError("Success cache is invalid") from error
            if candidate.concept_name_zh is not None and not validation:
                raise StandaloneError("Success cache lacks validation evidence")
        elif tech not in {"ready", "failed", row.skip_reason}:
            raise StandaloneError("State has an unknown technical status")


def _prepare_run(
    rows: Sequence[EnrichedRow],
    total_input_rows: int,
    input_csv: Path,
    semantic: dict[str, object],
    config: RuntimeConfig,
    concurrency: int,
    batch_size: int,
    resume_run_id: str | None,
    approved_pilot_run_id: str | None,
) -> tuple[Path, duckdb.DuckDBPyConnection]:
    """Create or verify one run manifest and its DuckDB state."""
    semantic_hash = _sha256_bytes(_canonical_json(semantic).encode("utf-8"))
    run_id = "run-" + semantic_hash[:16]
    if resume_run_id is not None and resume_run_id != run_id:
        raise StandaloneError("Resume run ID does not match current inputs")
    run_dir = run_directory(
        OUTPUT_ROOT,
        run_id,
        str(semantic.get("run_kind")),
        resume=resume_run_id is not None,
    )
    manifest_path = run_dir / "run_manifest.json"
    state_path = run_dir / "state.duckdb"
    if resume_run_id is None and run_dir.exists():
        raise StandaloneError(
            f"Run already exists; use --resume-run-id {run_id}"
        )
    if (
        resume_run_id is None
        and semantic.get("run_kind") == "pilot"
        and (OUTPUT_ROOT / run_id).exists()
    ):
        raise StandaloneError(
            f"Run already exists; use --resume-run-id {run_id}"
        )
    if resume_run_id is not None and not (
        manifest_path.is_file() and state_path.is_file()
    ):
        raise StandaloneError("Resume run is missing manifest or state")
    if resume_run_id is None:
        run_dir.mkdir(parents=True)
        manifest = {
            "run_id": run_id,
            "requested_model": config.model,
            "semantic": semantic,
            "input_path": str(input_csv.resolve()),
            "input_row_count": total_input_rows,
            "target_row_count": len(rows),
            "approved_pilot_run_id": approved_pilot_run_id,
            "skipped_existing_chinese_name": sum(
                row.skip_reason == "existing_chinese_name" for row in rows
            ),
            "user_confirmed_no_available_chinese_candidate": True,
            "premise_verified_by_script": False,
            "concurrency": concurrency,
            "batch_size": batch_size,
            "timeout_seconds": config.timeout_seconds,
        }
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
        os.replace(temporary, manifest_path)
        connection = duckdb.connect(str(state_path))
        try:
            _create_state(connection, rows, config, semantic_hash)
        except Exception:
            connection.close()
            raise
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("semantic") != semantic
            or manifest.get("input_row_count") != total_input_rows
            or manifest.get("target_row_count") != len(rows)
        ):
            raise StandaloneError("Run manifest does not match current inputs")
        connection = duckdb.connect(str(state_path))
        try:
            _check_resume(
                connection,
                rows,
                config,
                semantic_hash,
                str(semantic["base_url"]),
            )
        except Exception:
            connection.close()
            raise
    return run_dir, connection


def _persist_outcome(
    connection: duckdb.DuckDBPyConnection,
    row: EnrichedRow,
    request_fp: str,
    outcome: CallOutcome,
    model_name: str,
    thinking_mode: str | None,
    endpoint: str,
) -> None:
    """Commit response, validation, status, and call ledger atomically."""
    candidate, review = _decide_status(outcome)
    actual_model = outcome.model_name or model_name
    tech = "succeeded" if outcome.candidate is not None else "failed"
    response_json = (
        _canonical_json(outcome.candidate.model_dump(mode="json"))
        if outcome.candidate is not None
        else None
    )
    validation_json = (
        _canonical_json(outcome.validation.as_dict())
        if outcome.validation is not None
        else None
    )
    metadata = {
        "error_type": outcome.error_type,
        "http_status": outcome.http_status,
        "attempts": outcome.attempts,
        "latency_ms": outcome.latency_ms,
        "usage": outcome.usage,
        "call_id": outcome.call_id,
        "model_name": actual_model,
        "requested_model": model_name,
        "thinking_mode": thinking_mode,
        "endpoint": endpoint,
        "provider_error_code": outcome.provider_error_code,
    }
    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute(
            "UPDATE targets SET candidate_name_zh=?, review_status=?, "
            "tech_status=?, response_json=?, validation_json=?, "
            "metadata_json=?, model_name=?, request_fp=? WHERE seq=? "
            "AND tech_status <> 'succeeded'",
            [
                candidate,
                review,
                tech,
                response_json,
                validation_json,
                _canonical_json(metadata),
                actual_model,
                request_fp,
                row.input.seq,
            ],
        )
        connection.execute(
            "INSERT INTO calls (call_id, seq, request_fp, model_name, outcome, "
            "error_type, http_status, attempts, latency_ms, usage_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                outcome.call_id,
                row.input.seq,
                request_fp,
                actual_model,
                tech,
                outcome.error_type,
                outcome.http_status,
                outcome.attempts,
                outcome.latency_ms,
                _canonical_json(outcome.usage),
            ],
        )
        connection.execute("COMMIT")
    except (Exception, KeyboardInterrupt):
        connection.execute("ROLLBACK")
        raise


def _process_batches(
    connection: duckdb.DuckDBPyConnection,
    rows: Sequence[EnrichedRow],
    config: RuntimeConfig,
    concurrency: int,
    batch_size: int,
    pilot_only: bool,
    retry_auth_failures: bool = False,
    retry_prior_endpoint_bad_requests: bool = False,
    legacy_endpoint: str | None = None,
) -> str | None:
    """Commit each response, then stop dispatch on fatal provider errors."""
    del pilot_only  # Pilot rows are selected before state creation.
    work = [row for row in rows if row.skip_reason is None]
    saved = connection.execute(
        "SELECT seq, tech_status, metadata_json FROM targets"
    ).fetchall()
    technical = {
        int(seq): (str(status), metadata) for seq, status, metadata in saved
    }
    transient = {
        "quota_exhausted",
        "rate_limit",
        "network_timeout",
        "network_error",
        "request_timeout",
        "conflict",
        "too_early",
        "server_error",
    }
    if retry_auth_failures:
        transient.add("authentication_error")
    pending_work = []
    for row in work:
        status, metadata_json = technical[row.input.seq]
        error_type = None
        if metadata_json:
            metadata = json.loads(metadata_json)
            error_type = metadata.get("error_type")
        prior_endpoint = (
            metadata.get("endpoint", legacy_endpoint) if metadata_json else None
        )
        if (
            status == "ready"
            or (status == "failed" and error_type in transient)
            or (
                status == "failed"
                and error_type == "bad_request"
                and retry_prior_endpoint_bad_requests
                and prior_endpoint != config.base_url
            )
        ):
            pending_work.append(row)
    if not pending_work:
        return None
    chain = _build_chain(config)
    max_inflight = min(concurrency, batch_size)
    stop_on = {
        "quota_exhausted",
        "rate_limit",
        "authentication_error",
        "network_timeout",
        "network_error",
        "server_error",
    }
    stop_reason: str | None = None
    consecutive_bad_requests = 0
    processed_this_run = 0
    succeeded_this_run = 0
    with ThreadPoolExecutor(max_workers=max_inflight) as executor:
        for offset in range(0, len(pending_work), batch_size):
            batch = pending_work[offset : offset + batch_size]
            next_index = 0
            in_flight: dict[Future[CallOutcome], tuple[EnrichedRow, str]] = {}
            try:
                while next_index < len(batch) or in_flight:
                    while (
                        stop_reason is None
                        and next_index < len(batch)
                        and len(in_flight) < max_inflight
                    ):
                        row = batch[next_index]
                        request = _build_request(row)
                        fingerprint = _request_fingerprint(request, config)
                        future = executor.submit(
                            _invoke_with_retry, chain, request
                        )
                        in_flight[future] = (row, fingerprint)
                        next_index += 1
                    if not in_flight:
                        break
                    done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for future in done:
                        row, fingerprint = in_flight.pop(future)
                        outcome = future.result()
                        _persist_outcome(
                            connection,
                            row,
                            fingerprint,
                            outcome,
                            config.model,
                            config.thinking_mode,
                            config.base_url,
                        )
                        processed_this_run += 1
                        if outcome.error_type is None:
                            succeeded_this_run += 1
                        if outcome.error_type == "bad_request":
                            consecutive_bad_requests += 1
                            if consecutive_bad_requests >= 3:
                                stop_reason = "repeated_bad_request"
                        else:
                            consecutive_bad_requests = 0
                        if outcome.error_type in stop_on:
                            stop_reason = outcome.error_type
            except KeyboardInterrupt:
                for future, (row, fingerprint) in in_flight.items():
                    outcome = future.result()
                    _persist_outcome(
                        connection,
                        row,
                        fingerprint,
                        outcome,
                        config.model,
                        config.thinking_mode,
                        config.base_url,
                    )
                return "interrupted"
            print(
                "llm_trans: progress "
                f"{processed_this_run}/{len(pending_work)} processed, "
                f"{succeeded_this_run} succeeded, "
                f"{processed_this_run - succeeded_this_run} failed",
                file=sys.stderr,
                flush=True,
            )
            if stop_reason is not None:
                return stop_reason
    return None


def _export_and_verify(
    connection: duckdb.DuckDBPyConnection,
    run_dir: Path,
    rows: Sequence[EnrichedRow],
) -> Path:
    """Atomically export and verify the exact target CSV."""
    output = run_dir / "omop-trans.csv"
    temporary = run_dir / ".omop-trans.csv.tmp"
    records = connection.execute(
        "SELECT concept_id, original_name_zh, concept_name, domain_id, "
        "vocabulary_id, concept_class_id, candidate_name_zh, "
        "review_status, coalesce(model_name, '') FROM targets ORDER BY seq"
    ).fetchall()
    if len(records) != len(rows):
        raise StandaloneError("Export target count does not match input")
    expected_ids = [row.input.values[0] for row in rows]
    if [record[0] for record in records] != expected_ids:
        raise StandaloneError("Export target order or IDs do not match")
    allowed = {"pending_expert", "not_required", "approved", "rejected"}
    for record in records:
        if record[7] not in allowed:
            raise StandaloneError("Export contains an unknown review status")
        if record[7] == "pending_expert" and (
            record[1] or not record[6] or not record[8]
        ):
            raise StandaloneError("Pending candidate has invalid columns")
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(
            destination,
            delimiter="~",
            lineterminator="\n",
        )
        writer.writerow(OUTPUT_COLUMNS)
        writer.writerows(records)
        destination.flush()
        os.fsync(destination.fileno())
    with temporary.open("r", encoding="utf-8", newline="") as source:
        reader = csv.reader(source, delimiter="~", strict=True)
        if tuple(next(reader, ())) != OUTPUT_COLUMNS:
            raise StandaloneError("Export header verification failed")
        verified = list(reader)
    if (
        len(verified) != len(rows)
        or [record[0] for record in verified] != expected_ids
        or len(set(expected_ids)) != len(expected_ids)
        or any(len(record) != len(OUTPUT_COLUMNS) for record in verified)
    ):
        raise StandaloneError("Export row verification failed")
    os.replace(temporary, output)
    return output


def _validated_pilot_states(full_semantic: Mapping[str, object]) -> list[Path]:
    """Find complete, non-failing Pilot runs for the same source snapshot."""
    if not OUTPUT_ROOT.is_dir():
        return []
    expected = {
        key: value
        for key, value in full_semantic.items()
        if key not in {"run_kind", "pilot_size", "pilot_label", "script_sha256"}
    }
    valid: list[Path] = []
    for run_dir in pilot_directories(OUTPUT_ROOT):
        manifest_path = run_dir / "run_manifest.json"
        state_path = run_dir / "state.duckdb"
        output_path = run_dir / "omop-trans.csv"
        if not (
            manifest_path.is_file()
            and state_path.is_file()
            and output_path.is_file()
        ):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            semantic = manifest["semantic"]
            if semantic.get("run_kind") != "pilot":
                continue
            matching = {
                key: value
                for key, value in semantic.items()
                if key
                not in {
                    "run_kind",
                    "pilot_size",
                    "pilot_label",
                    "script_sha256",
                }
            }
            if matching != expected:
                continue
            with output_path.open(encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source, delimiter="~")
                if tuple(reader.fieldnames or ()) != OUTPUT_COLUMNS:
                    continue
                exported = list(reader)
            pilot = duckdb.connect(str(state_path), read_only=True)
            try:
                records = pilot.execute(
                    "SELECT concept_id, candidate_name_zh, tech_status, "
                    "validation_json, response_json, model_name "
                    "FROM targets ORDER BY seq"
                ).fetchall()
            finally:
                pilot.close()
            if not records or len(records) != len(exported):
                continue
            accepted = True
            for record, (
                concept_id,
                candidate,
                tech,
                validation,
                response,
                model,
            ) in zip(exported, records, strict=True):
                if (
                    tech != "succeeded"
                    or not candidate
                    or not validation
                    or not response
                    or not model
                    or record["concept_id"] != concept_id
                    or record["candidate_name_zh"] != candidate
                    or record["model_name"] != model
                ):
                    accepted = False
                    break
                parsed = LlmCandidateResponse.model_validate(
                    _load_strict_json(response)
                )
                if parsed.concept_name_zh != candidate or json.loads(
                    validation
                ).get("validation_status") not in {"pass", "review"}:
                    accepted = False
                    break
            if accepted:
                valid.append(state_path)
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            duckdb.Error,
            ValidationError,
        ):
            continue
    return valid


def _seed_pilot_successes(
    connection: duckdb.DuckDBPyConnection,
    pilot_states: Sequence[Path],
    rows: Sequence[EnrichedRow],
    config: RuntimeConfig,
) -> None:
    """Copy checked Pilot responses into the full state without new requests."""
    by_seq = {row.input.seq: row for row in rows}
    connection.execute("BEGIN TRANSACTION")
    try:
        for state_path in pilot_states:
            pilot = duckdb.connect(str(state_path), read_only=True)
            try:
                records = pilot.execute(
                    "SELECT seq, enriched_json, request_fp, candidate_name_zh, "
                    "review_status, response_json, validation_json, "
                    "metadata_json, model_name FROM targets "
                    "WHERE tech_status='succeeded'"
                ).fetchall()
            finally:
                pilot.close()
            for (
                seq,
                enriched,
                fingerprint,
                candidate,
                review,
                response,
                validation,
                metadata_json,
                model,
            ) in records:
                target = connection.execute(
                    "SELECT enriched_json, tech_status, response_json, "
                    "model_name FROM targets WHERE seq=?",
                    [seq],
                ).fetchone()
                if target is None or target[0] != enriched:
                    raise StandaloneError(
                        "Pilot target identity does not match"
                    )
                if target[1] == "succeeded":
                    if target[2] != response or target[3] != model:
                        raise StandaloneError(
                            "Conflicting Pilot success for one concept"
                        )
                    continue
                metadata = json.loads(metadata_json)
                requested = metadata.get("requested_model")
                thinking = metadata.get("thinking_mode")
                if (
                    not isinstance(requested, str)
                    or not requested
                    or thinking is not None
                    and not isinstance(thinking, str)
                ):
                    raise StandaloneError("Pilot model provenance is invalid")
                used_config = dataclass_replace(
                    config, model=requested, thinking_mode=thinking
                )
                if fingerprint != _request_fingerprint(
                    _build_request(by_seq[seq]), used_config
                ):
                    raise StandaloneError(
                        "Pilot request fingerprint does not match current code"
                    )
                metadata["seeded_from_run_id"] = state_path.parent.name
                connection.execute(
                    "UPDATE targets SET request_fp=?, candidate_name_zh=?, "
                    "review_status=?, tech_status='succeeded', "
                    "response_json=?, validation_json=?, metadata_json=?, "
                    "model_name=? WHERE seq=? AND tech_status='ready'",
                    [
                        fingerprint,
                        candidate,
                        review,
                        response,
                        validation,
                        _canonical_json(metadata),
                        model,
                        seq,
                    ],
                )
        connection.execute("COMMIT")
    except (Exception, KeyboardInterrupt):
        connection.execute("ROLLBACK")
        raise


def llm_trans(
    input_csv: Path,
    vocabulary_id: str,
    concurrency: int = 1,
    batch_size: int = 20,
    *,
    resume_run_id: str | None = None,
    pilot_only: bool = False,
    pilot_size: int = 3,
    pilot_label: str = "default",
    model_name: str | None = None,
    require_pilot: bool = False,
    approved_pilot_run_id: str | None = None,
    allow_code_hash_drift: bool = False,
    allow_endpoint_switch: bool = False,
    retry_auth_failures: bool = False,
) -> Path:
    """Translate one vocabulary into review-only candidates.

    Args:
        input_csv: Strict UTF-8, tilde-delimited seven-column source.
        vocabulary_id: Exact vocabulary value in the input CSV.
        concurrency: Maximum simultaneous external requests.
        batch_size: Maximum target requests in one processing batch.
        resume_run_id: Existing semantic run ID to continue, if any.
        pilot_only: Create a separate sampled Pilot run.
        pilot_size: Maximum 1..20 sampled eligible targets.
        pilot_label: Separate identity for a new Pilot attempt.
        model_name: Optional model override for this invocation.
        require_pilot: Require a validated Pilot before a new full run.
        approved_pilot_run_id: Human-inspected Pilot to reuse.
        allow_code_hash_drift: Explicitly audit a code-only resume change.
        allow_endpoint_switch: Require a current-endpoint Pilot for a move.
        retry_auth_failures: Probe the endpoint, then retry prior auth errors.

    Returns:
        Verified path to the nine-column CSV.

    Raises:
        StandaloneError: If preflight, identity, or state checks fail.
    """
    if not vocabulary_id.strip():
        raise StandaloneError("vocabulary_id must be nonempty")
    if retry_auth_failures and resume_run_id is None:
        raise StandaloneError("--retry-auth-failures requires --resume-run-id")
    if vocabulary_id.strip().casefold() in (
        {value.casefold() for value in EXTERNAL_LLM_DENIED_VOCABULARY_IDS}
        | INVALID_VOCABULARY_SENTINELS
    ):
        raise StandaloneError(
            "External LLM transfer is denied for this vocabulary_id"
        )
    if (
        type(concurrency) is not int
        or concurrency <= 0
        or type(batch_size) is not int
        or batch_size <= 0
    ):
        raise StandaloneError("concurrency and batch_size must be positive")
    if type(pilot_size) is not int or not 1 <= pilot_size <= 20:
        raise StandaloneError("pilot_size must be in 1..20")
    if re.fullmatch(r"[A-Za-z0-9_-]{1,40}", pilot_label) is None:
        raise StandaloneError("pilot_label must use 1..40 safe characters")
    if pilot_label != "default" and not pilot_only:
        raise StandaloneError("pilot_label is only for Pilot runs")
    if pilot_only and approved_pilot_run_id is not None:
        raise StandaloneError("Pilot cannot approve itself")
    if (
        approved_pilot_run_id is not None
        and re.fullmatch(r"run-[0-9a-f]{16}", approved_pilot_run_id) is None
    ):
        raise StandaloneError("Invalid approved Pilot run ID")
    config = _load_runtime()
    if model_name is not None:
        if not model_name.strip() or re.search(r"[\x00-\x1f]", model_name):
            raise StandaloneError("model_name must be nonempty plain text")
        config = dataclass_replace(config, model=model_name)
    input_csv = Path(input_csv).expanduser().resolve()
    selected, total, input_hash = _validate_input(
        input_csv,
        vocabulary_id,
    )
    hashes = _verify_athena(config)
    rows = _enrich_targets(selected, config)
    if pilot_only:
        eligible = [row for row in rows if row.skip_reason is None]
        count = min(pilot_size, len(eligible))
        rows = (
            [eligible[i * len(eligible) // count] for i in range(count)]
            if count
            else []
        )
    semantic = _semantic_manifest(
        input_hash,
        vocabulary_id,
        hashes,
        config,
    )
    semantic["run_kind"] = "pilot" if pilot_only else "full"
    if pilot_only:
        semantic["pilot_size"] = pilot_size
        semantic["pilot_label"] = pilot_label
    pilot_sources: list[Path] = []
    if not pilot_only and resume_run_id is None:
        pilot_sources = _validated_pilot_states(semantic)
        if approved_pilot_run_id is not None:
            pilot_sources = [
                path
                for path in pilot_sources
                if path.parent.name == approved_pilot_run_id
            ]
            if not pilot_sources:
                raise StandaloneError(
                    "Approved Pilot is missing, incompatible, or failed"
                )
        if len(pilot_sources) > 1:
            raise StandaloneError(
                "Multiple validated Pilots; specify --approved-pilot-run-id"
            )
        if (
            require_pilot
            and any(row.skip_reason is None for row in rows)
            and not pilot_sources
        ):
            raise StandaloneError(
                "No validated Pilot exists for this vocabulary"
            )
    if resume_run_id is not None and allow_endpoint_switch:
        valid_pilots = _validated_pilot_states(semantic)
        if approved_pilot_run_id is None or not any(
            path.parent.name == approved_pilot_run_id for path in valid_pilots
        ):
            raise StandaloneError(
                "Endpoint switch requires a validated --approved-pilot-run-id"
            )
    semantic = _resolve_resume_semantic(
        semantic,
        resume_run_id,
        allow_code_hash_drift,
        allow_endpoint_switch,
    )
    semantic_hash = _sha256_bytes(_canonical_json(semantic).encode("utf-8"))
    run_id = "run-" + semantic_hash[:16]
    output_root = OUTPUT_ROOT
    output_root.mkdir(exist_ok=True)
    lock = FileLock(str(output_root / f".{run_id}.lock"), timeout=0)
    try:
        with lock:
            run_dir, connection = _prepare_run(
                rows,
                total,
                input_csv,
                semantic,
                config,
                concurrency,
                batch_size,
                resume_run_id,
                pilot_sources[0].parent.name if pilot_sources else None,
            )
            try:
                rebuild_index(
                    output_root,
                    live_run_id=run_id,
                    live_connection=connection,
                )
                if not pilot_only and resume_run_id is None and pilot_sources:
                    _seed_pilot_successes(
                        connection, pilot_sources, rows, config
                    )
                if retry_auth_failures:
                    probe = _probe_model(config)
                    _record_model_probe(
                        input_csv,
                        vocabulary_id,
                        pilot_only,
                        config.model,
                        probe,
                    )
                    if probe.error_type is not None:
                        raise StandaloneError(
                            "Auth retry probe failed: " + probe.error_type
                        )
                interrupted = _process_batches(
                    connection,
                    rows,
                    config,
                    concurrency,
                    batch_size,
                    pilot_only,
                    retry_auth_failures,
                    allow_endpoint_switch,
                    str(semantic["base_url"]),
                )
                output = _export_and_verify(
                    connection,
                    run_dir,
                    rows,
                )
            finally:
                connection.close()
                rebuild_index(output_root)
            if interrupted == "interrupted":
                raise KeyboardInterrupt
            if interrupted is not None:
                raise _RunStopped(interrupted, output)
            return output
    except Timeout as error:
        raise StandaloneError("Another process is writing this run") from error


class _RunStopped(StandaloneError):
    """Preserve the exported checkpoint path for a stopped model."""

    def __init__(self, reason: str, output: Path) -> None:
        super().__init__(
            f"Stopped after {reason}; committed results remain resumable: "
            f"{output}"
        )
        self.reason = reason
        self.output = output


def _probe_model(config: RuntimeConfig) -> CallOutcome:
    """Verify a new model with synthetic terminology before real transfer."""
    request = LlmConceptRequest(
        concept_id=0,
        concept_name="Synthetic left kidney",
        concept_code="SYNTHETIC-ONLY",
        vocabulary_id="SYNTHETIC",
        domain_id="Condition",
        concept_class_id="Clinical Finding",
        input_fingerprint="synthetic-probe",
    )
    return _invoke_with_retry(_build_chain(config), request)


def _record_model_probe(
    input_csv: Path,
    vocabulary_id: str,
    pilot_only: bool,
    requested_model: str,
    outcome: CallOutcome,
) -> None:
    """Append a secret-free usage record for a synthetic availability probe."""
    OUTPUT_ROOT.mkdir(exist_ok=True)
    record = {
        "input_sha256": _sha256_file(input_csv),
        "vocabulary_id": vocabulary_id,
        "run_kind": "pilot" if pilot_only else "full",
        "requested_model": requested_model,
        "model_name": outcome.model_name or requested_model,
        "error_type": outcome.error_type,
        "attempts": outcome.attempts,
        "usage": outcome.usage,
        "time_utc": datetime.now(UTC).isoformat(),
    }
    with FileLock(str(OUTPUT_ROOT / ".model-probes.lock")):
        with (OUTPUT_ROOT / "model-probes.jsonl").open(
            "a", encoding="utf-8", newline=""
        ) as target:
            target.write(_canonical_json(record) + "\n")
            target.flush()
            os.fsync(target.fileno())


def llm_trans_with_fallback(
    input_csv: Path,
    vocabulary_id: str,
    models: Sequence[str],
    *,
    concurrency: int = 1,
    batch_size: int = 20,
    resume_run_id: str | None = None,
    pilot_only: bool = False,
    pilot_size: int = 3,
    pilot_label: str = "default",
    require_pilot: bool = False,
    approved_pilot_run_id: str | None = None,
    allow_code_hash_drift: bool = False,
    allow_endpoint_switch: bool = False,
    retry_auth_failures: bool = False,
) -> Path:
    """Continue on quota exhaustion only after a synthetic model probe."""
    if not models or len(set(models)) != len(models):
        raise StandaloneError("Provide distinct models in preference order")
    current_run_id = resume_run_id
    for index, model in enumerate(models):
        if index:
            config = dataclass_replace(_load_runtime(), model=model)
            probe = _probe_model(config)
            _record_model_probe(
                Path(input_csv), vocabulary_id, pilot_only, model, probe
            )
            if probe.error_type is not None:
                if probe.error_type == "quota_exhausted":
                    continue
                raise StandaloneError(
                    f"Fallback model probe failed: {probe.error_type}"
                )
        try:
            return llm_trans(
                input_csv,
                vocabulary_id,
                concurrency=concurrency,
                batch_size=batch_size,
                resume_run_id=current_run_id,
                pilot_only=pilot_only,
                pilot_size=pilot_size,
                pilot_label=pilot_label,
                model_name=model,
                require_pilot=require_pilot,
                approved_pilot_run_id=approved_pilot_run_id,
                allow_code_hash_drift=allow_code_hash_drift,
                allow_endpoint_switch=allow_endpoint_switch,
                retry_auth_failures=retry_auth_failures,
            )
        except _RunStopped as error:
            if error.reason != "quota_exhausted":
                raise
            current_run_id = error.output.parent.name
    raise StandaloneError(
        "All listed models exhausted; committed results remain resumable"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the standalone translation CLI and return a process exit code."""
    parser = argparse.ArgumentParser(
        description="Generate review-only OMOP Chinese candidates."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    translate = commands.add_parser("translate")
    translate.add_argument("--input", type=Path, required=True)
    translate.add_argument("--vocabulary-id", required=True)
    translate.add_argument("--concurrency", type=int, default=1)
    translate.add_argument("--batch-size", type=int, default=20)
    translate.add_argument("--resume-run-id")
    translate.add_argument("--allow-code-hash-drift", action="store_true")
    translate.add_argument("--allow-endpoint-switch", action="store_true")
    translate.add_argument("--retry-auth-failures", action="store_true")
    translate.add_argument("--pilot-only", action="store_true")
    translate.add_argument("--pilot-size", type=int, default=3)
    translate.add_argument("--pilot-label", default="default")
    translate.add_argument("--model-name")
    translate.add_argument("--require-pilot", action="store_true")
    translate.add_argument("--approved-pilot-run-id")
    translate.add_argument("--fallback-model", action="append", default=[])
    report = commands.add_parser("report")
    report.add_argument("--input", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            from llm_omop_trans.report import build_report

            merged, report_path = build_report(args.input)
            print(merged)
            print(report_path)
            return 0
        if args.fallback_model:
            primary = args.model_name or _load_runtime().model
            output = llm_trans_with_fallback(
                args.input,
                args.vocabulary_id,
                [primary, *args.fallback_model],
                concurrency=args.concurrency,
                batch_size=args.batch_size,
                resume_run_id=args.resume_run_id,
                pilot_only=args.pilot_only,
                pilot_size=args.pilot_size,
                pilot_label=args.pilot_label,
                require_pilot=args.require_pilot,
                approved_pilot_run_id=args.approved_pilot_run_id,
                allow_code_hash_drift=args.allow_code_hash_drift,
                allow_endpoint_switch=args.allow_endpoint_switch,
                retry_auth_failures=args.retry_auth_failures,
            )
        else:
            output = llm_trans(
                args.input,
                args.vocabulary_id,
                concurrency=args.concurrency,
                batch_size=args.batch_size,
                resume_run_id=args.resume_run_id,
                pilot_only=args.pilot_only,
                pilot_size=args.pilot_size,
                pilot_label=args.pilot_label,
                model_name=args.model_name,
                require_pilot=args.require_pilot,
                approved_pilot_run_id=args.approved_pilot_run_id,
                allow_code_hash_drift=args.allow_code_hash_drift,
                allow_endpoint_switch=args.allow_endpoint_switch,
                retry_auth_failures=args.retry_auth_failures,
            )
    except StandaloneError as error:
        print(f"llm_trans: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "llm_trans: interrupted; committed results remain resumable",
            file=sys.stderr,
        )
        return 130
    print(output)
    return 0
