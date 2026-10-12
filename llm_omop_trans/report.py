"""Summarize isolated standalone runs without promoting LLM candidates."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

import duckdb

from llm_omop_trans.config import (
    EXTERNAL_LLM_DENIED_VOCABULARY_IDS,
    INPUT_COLUMNS,
    INVALID_VOCABULARY_SENTINELS,
    OUTPUT_COLUMNS,
    OUTPUT_ROOT,
    StandaloneError,
)
from llm_omop_trans.run_index import pilot_directories
from llm_omop_trans.utils import _sha256_file

TASK_EXCLUDED = EXTERNAL_LLM_DENIED_VOCABULARY_IDS | {
    "dm+d",
    "NDC",
} | INVALID_VOCABULARY_SENTINELS


def _has_chinese(value: str) -> bool:
    return any("\u3400" <= char <= "\u9fff" for char in value)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write verified UTF-8 tilde CSV using the canonical field order."""
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=OUTPUT_COLUMNS,
            delimiter="~",
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def _load_input_counts(
    input_csv: Path,
) -> tuple[Counter[str], dict[str, set[str]], list[str]]:
    counts: Counter[str] = Counter()
    input_ids: dict[str, set[str]] = defaultdict(set)
    invalid_ids: list[str] = []
    with input_csv.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="~")
        if tuple(reader.fieldnames or ()) != INPUT_COLUMNS:
            raise StandaloneError("Input CSV header does not match")
        seen: set[str] = set()
        for row in reader:
            if not row or None in row or None in row.values():
                raise StandaloneError("Input CSV has malformed records")
            concept_id = row["concept_id"]
            if concept_id in seen:
                raise StandaloneError("Input CSV repeats a concept_id")
            seen.add(concept_id)
            vocabulary = row["vocabulary_id"]
            if vocabulary.strip().casefold() in INVALID_VOCABULARY_SENTINELS:
                invalid_ids.append(concept_id)
            elif vocabulary not in TASK_EXCLUDED:
                counts[vocabulary] += 1
                input_ids[vocabulary].add(concept_id)
    return counts, input_ids, invalid_ids


def _load_runs(
    output_root: Path,
    input_hash: str,
) -> tuple[dict[str, Path], list[tuple[str, Path]]]:
    full: dict[str, Path] = {}
    all_runs: list[tuple[str, Path]] = []
    for run_dir in pilot_directories(output_root):
        manifest_path = run_dir / "run_manifest.json"
        state_path = run_dir / "state.duckdb"
        if not (manifest_path.is_file() and state_path.is_file()):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            semantic = manifest["semantic"]
            kind = semantic.get("run_kind")
            vocabulary = semantic["vocabulary_id"]
            if (
                semantic.get("input_sha256") != input_hash
                or kind not in {"pilot", "full"}
                or vocabulary in TASK_EXCLUDED
            ):
                continue
        except (OSError, ValueError, KeyError, TypeError):
            continue
        all_runs.append((kind, run_dir))
        if kind == "full":
            if vocabulary in full:
                raise StandaloneError(
                    f"Multiple full runs for vocabulary_id {vocabulary}"
                )
            full[vocabulary] = run_dir
    return full, all_runs


def build_report(
    input_csv: Path,
    output_root: Path = OUTPUT_ROOT,
) -> tuple[Path, Path]:
    """Merge checked full-run exports and report incomplete work explicitly."""
    input_csv = input_csv.expanduser().resolve()
    output_root.mkdir(exist_ok=True)
    counts, input_ids, invalid_ids = _load_input_counts(input_csv)
    input_hash = _sha256_file(input_csv)
    full_runs, all_runs = _load_runs(output_root, input_hash)
    merged: list[dict[str, str]] = []
    seen: set[str] = set()
    summary: list[dict[str, object]] = []
    per_model: dict[str, Counter[str]] = defaultdict(Counter)
    quota_events: list[str] = []
    probe_events: list[str] = []
    switch_events: list[str] = []
    failure_events: list[str] = []
    failed_checks = 0
    review_count = 0
    no_candidate_count = 0
    issue_summary: list[tuple[str, str, int, int, int, Path]] = []
    for kind, run_dir in all_runs:
        state = duckdb.connect(str(run_dir / "state.duckdb"), read_only=True)
        try:
            calls = state.execute(
                "SELECT model_name, error_type, attempts, usage_json "
                "FROM calls ORDER BY completed_at, call_id"
            ).fetchall()
            errors = Counter(str(error) for _, error, _, _ in calls if error)
            for error, count in sorted(errors.items()):
                failure_events.append(
                    f"{run_dir.name} ({kind}): {error} × {count}"
                )
            if kind == "full":
                sequence: list[str] = []
                for model, *_ in calls:
                    if not sequence or sequence[-1] != model:
                        sequence.append(str(model))
                if len(sequence) > 1:
                    switch_events.append(
                        f"{run_dir.name}: " + " -> ".join(sequence)
                    )
            for model, error, attempts, usage_json in calls:
                stats = per_model[str(model)]
                stats["request_attempts"] += attempts
                stats["call_records"] += 1
                usage = json.loads(usage_json)
                for key in (
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                ):
                    if usage.get(key) is None:
                        stats[key + "_unknown"] += 1
                    else:
                        stats[key] += usage[key]
                if error == "quota_exhausted":
                    quota_events.append(
                        f"{run_dir.name} ({kind}): {model} quota_exhausted"
                    )
        finally:
            state.close()

    probe_path = output_root / "model-probes.jsonl"
    if probe_path.is_file():
        for line in probe_path.read_text(encoding="utf-8").splitlines():
            try:
                probe = json.loads(line)
                if (
                    probe.get("input_sha256") != input_hash
                    or probe.get("vocabulary_id") not in counts
                ):
                    continue
                model = str(probe["model_name"])
                attempts = int(probe["attempts"])
                usage = probe["usage"]
                error = probe["error_type"]
                stats = per_model[model]
                stats["request_attempts"] += attempts
                stats["call_records"] += 1
                stats["probe_requests"] += attempts
                for key in (
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                ):
                    if usage.get(key) is None:
                        stats[key + "_unknown"] += 1
                    else:
                        stats[key] += usage[key]
                probe_events.append(
                    f"{probe['vocabulary_id']} / {probe['run_kind']}: "
                    f"{probe['requested_model']} -> "
                    f"{'可用' if error is None else error}"
                )
                if error == "quota_exhausted":
                    quota_events.append(
                        f"synthetic probe: {model} quota_exhausted"
                    )
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                raise StandaloneError(
                    "Model probe ledger is invalid"
                ) from error

    for vocabulary, total in sorted(counts.items()):
        full_dir = full_runs.get(vocabulary)
        if full_dir is None:
            summary.append(
                {
                    "vocabulary": vocabulary,
                    "total": total,
                    "processed": 0,
                    "api_success": 0,
                    "candidate": 0,
                    "postcheck_pass": 0,
                    "postcheck_review": 0,
                    "postcheck_fail": 0,
                    "failed": 0,
                    "pending": total,
                    "skipped": 0,
                    "run": "—",
                    "status": "未启动",
                }
            )
            continue
        output_path = full_dir / "omop-trans.csv"
        if not output_path.is_file():
            raise StandaloneError(f"Full run lacks CSV: {full_dir.name}")
        with output_path.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source, delimiter="~")
            if tuple(reader.fieldnames or ()) != OUTPUT_COLUMNS:
                raise StandaloneError(f"Bad CSV columns: {full_dir.name}")
            rows = list(reader)
        if (
            len(rows) != total
            or {row["concept_id"] for row in rows} != input_ids[vocabulary]
        ):
            raise StandaloneError(
                f"Input/CSV row-count mismatch: {full_dir.name}"
            )
        state = duckdb.connect(str(full_dir / "state.duckdb"), read_only=True)
        try:
            records = state.execute(
                "SELECT concept_id, candidate_name_zh, review_status, "
                "tech_status, validation_json, model_name FROM targets "
                "ORDER BY seq"
            ).fetchall()
        finally:
            state.close()
        if len(records) != total:
            raise StandaloneError(f"Input/state mismatch: {full_dir.name}")
        metrics: Counter[str] = Counter()
        postcheck_fail: list[dict[str, str]] = []
        needs_review: list[dict[str, str]] = []
        no_candidate: list[dict[str, str]] = []
        for row, (
            concept_id,
            candidate,
            review,
            tech,
            validation,
            model,
        ) in zip(rows, records, strict=True):
            if (
                row["concept_id"] != concept_id
                or row["candidate_name_zh"] != candidate
                or row["review_status"] != review
                or row["model_name"] != (model or "")
                or row["vocabulary_id"] != vocabulary
            ):
                raise StandaloneError(f"CSV/state mismatch: {full_dir.name}")
            if concept_id in seen:
                raise StandaloneError(f"Duplicate concept_id: {concept_id}")
            seen.add(concept_id)
            merged.append(row)
            if tech == "succeeded":
                metrics["api_success"] += 1
            elif tech == "failed":
                metrics["failed"] += 1
                metrics["api_failed"] += 1
            elif tech == "concept_invalid":
                metrics["failed"] += 1
                metrics["precheck_failed"] += 1
            elif tech == "ready":
                metrics["pending"] += 1
            elif tech == "existing_chinese_name":
                metrics["skipped"] += 1
            else:
                metrics["failed"] += 1
            if candidate:
                metrics["candidate"] += 1
            if validation:
                status = json.loads(validation).get("validation_status")
                if status not in {"pass", "review", "fail"}:
                    raise StandaloneError("Unknown postcheck status")
                metrics["postcheck_" + status] += 1
                if status == "fail":
                    postcheck_fail.append(row)
            if review == "pending_expert" or (validation and status == "fail"):
                needs_review.append(row)
            if not _has_chinese(candidate) and not row["concept_name_zh"]:
                no_candidate.append(row)
        for filename, issue_rows in (
            ("postcheck-fail.csv", postcheck_fail),
            ("needs-review.csv", needs_review),
            ("no-candidate.csv", no_candidate),
        ):
            _write_csv(full_dir / filename, issue_rows)
        failed_checks += len(postcheck_fail)
        review_count += len(needs_review)
        no_candidate_count += len(no_candidate)
        issue_summary.append(
            (
                vocabulary,
                full_dir.name,
                len(postcheck_fail),
                len(needs_review),
                len(no_candidate),
                full_dir,
            )
        )
        processed = total - metrics["pending"]
        completed = total - metrics["pending"] - metrics["failed"]
        status = (
            "完成"
            if metrics["pending"] == 0 and metrics["failed"] == 0
            else "部分完成"
        )
        summary.append(
            {
                "vocabulary": vocabulary,
                "total": total,
                "processed": processed,
                "api_success": metrics["api_success"],
                "candidate": metrics["candidate"],
                "postcheck_pass": metrics["postcheck_pass"],
                "postcheck_review": metrics["postcheck_review"],
                "postcheck_fail": metrics["postcheck_fail"],
                "failed": metrics["failed"],
                "api_failed": metrics["api_failed"],
                "precheck_failed": metrics["precheck_failed"],
                "pending": metrics["pending"],
                "skipped": metrics["skipped"],
                "completed": completed,
                "run": full_dir.name,
                "status": status,
            }
        )
    merged_path = output_root / "omop-trans-all.csv"
    _write_csv(merged_path, merged)
    with merged_path.open(encoding="utf-8", newline="") as source:
        verified = list(csv.DictReader(source, delimiter="~"))
    if len(verified) != len(merged) or len(seen) != len(merged):
        raise StandaloneError("Merged CSV verification failed")
    report_path = output_root / "llm-trans-report.md"
    api_failed_total = sum(
        cast(int, item.get("api_failed", 0)) for item in summary
    )
    precheck_failed_total = sum(
        cast(int, item.get("precheck_failed", 0)) for item in summary
    )
    lines = [
        "# OMOP 独立翻译运行报告",
        "",
        f"输入：`{input_csv}`；输出：`{merged_path}`。",
        "",
        "本报告仅统计本版独立入口的 Pilot 和正式运行。API 成功、非空候选、"
        "后检通过和全部处理完成是不同指标。正式中文名未自动接受。",
        "",
        "## 词表进度",
        "",
        "| vocabulary_id | 输入 | 已处理 | API 成功 | 非空候选 | "
        "后检 pass/review/fail "
        "| 失败（含预检） | 待请求 | 跳过 | 完成率 | run-id | 状态 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
        "| --- | --- |",
    ]
    for item in summary:
        total = cast(int, item["total"])
        completed = cast(int, item.get("completed", 0))
        rate = f"{100 * completed / total:.2f}%" if total else "0.00%"
        lines.append(
            f"| {item['vocabulary']} | {total} | {item['processed']} | "
            f"{item['api_success']} "
            f"| {item['candidate']} | {item['postcheck_pass']}/"
            f"{item['postcheck_review']}/{item['postcheck_fail']} "
            f"| {item['failed']} | {item['pending']} | {item['skipped']} "
            f"| {rate} | {item['run']} | {item['status']} |"
        )
    lines += [
        "",
        "完成率按输入行中不再待请求且未失败的比例计算；数据预检失败"
        "计入失败，已有中文名的跳过项单列显示，"
        "不算 API 成功。各 run 的原始 CSV 位于 "
        "`outputs/<run-id>/omop-trans.csv`。",
        "",
        "## 逐词表文件与异常清单",
        "",
        "| vocabulary_id | run-id | 正式 CSV | 后检失败 | 文件 | "
        "需人工复核 | 文件 | 无中文候选 | 文件 |",
        "| --- | --- | --- | ---: | --- | ---: | --- | ---: | --- |",
    ]
    for vocabulary, run_id, fail, review, empty, directory in issue_summary:
        lines.append(
            f"| {vocabulary} | {run_id} | `{directory / 'omop-trans.csv'}` "
            f"| {fail} | `{directory / 'postcheck-fail.csv'}` "
            f"| {review} | `{directory / 'needs-review.csv'}` "
            f"| {empty} | `{directory / 'no-candidate.csv'}` |"
        )
    lines += [
        "",
        "## 模型调用与 Token",
        "",
        "| model_name | 请求尝试 | 其中探测 | 调用记录 | prompt tokens | "
        "completion tokens | total tokens | 用量未知记录 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, stats in sorted(per_model.items()):
        lines.append(
            f"| {model} | {stats['request_attempts']} | "
            f"{stats['probe_requests']} | {stats['call_records']} "
            f"| {stats['prompt_tokens']} | {stats['completion_tokens']} "
            f"| {stats['total_tokens']} | {stats['total_tokens_unknown']} |"
        )
    lines += [
        "",
        "Token 仅为 API 响应中实际返回并持久化的用量；失败重试和无用量响应"
        "无法推算。额度剩余值若 API 未提供，则不可得。",
        "",
        "额度耗尽记录：",
    ]
    lines += [f"- {event}" for event in quota_events] or ["- 无已记录事件"]
    lines += ["", "正式运行模型切换："]
    lines += [f"- {event}" for event in switch_events] or ["- 无已记录事件"]
    lines += ["", "模型切换前的合成探测："]
    lines += [f"- {event}" for event in probe_events] or ["- 无已记录事件"]
    lines += ["", "请求异常记录："]
    lines += [f"- {event}" for event in failure_events] or ["- 无已记录事件"]
    lines += [
        "",
        "## 完整性与异常",
        "",
        f"- 正式运行已导出 {len(merged)} 行；`concept_id` "
        f"唯一数 {len(seen)}，CSV 与状态库逐行一致。",
        f"- 后检失败 {failed_checks} 行；需人工复核 {review_count} 行；"
        f"无中文候选 {no_candidate_count} 行。",
        f"- 其中请求失败 {api_failed_total} 行；"
        f"输入预检失败 {precheck_failed_total} 行。",
        f"- 无效 vocabulary_id 占位行 {len(invalid_ids)} 条"
        f"（concept_id: {', '.join(invalid_ids) or '无'}）；未外发、未合并。",
        "- 对应文件在各 run 目录：`postcheck-fail.csv`、"
        "`needs-review.csv`、`no-candidate.csv`。",
        f"- 待启动或未完成词表："
        f" {sum(item['status'] != '完成' for item in summary)} 个。",
        "- 下一步：按 `docs/Permit.md` 的本次授权继续逐词表 Pilot 质检。"
        "确认 API 端点恢复后按断点续跑；对后检失败和无中文候选进行人工审核。",
        "",
    ]
    temporary = report_path.with_name("." + report_path.name + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, report_path)
    return merged_path, report_path
