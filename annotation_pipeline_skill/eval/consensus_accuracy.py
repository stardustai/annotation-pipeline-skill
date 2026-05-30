"""Offline accuracy evaluation with multi-pass consensus judging.

The LLM judge used for accuracy QC (the ``qc`` LLM target) is stochastic: the
same span may be flagged as an error on one pass and not the next. A single
pass over a random sample therefore yields a noisy precision/recall/F1 estimate
that can swing several points run-to-run (empirically, the same task set has
flagged 41 vs 2 false positives across two runs).

This module runs the judge ``passes`` times over the SAME rows and keeps only
errors confirmed by a *majority* of passes, filtering out transient
hallucinations to recover the true-quality estimate.

The judge is dependency-injected (``JudgeFn``) so the scoring logic is fully
testable without network access. ``build_qc_judge`` constructs the production
judge from a configured :class:`LLMProfile` (resolved from the ``qc`` target).
"""

from __future__ import annotations

import collections
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Sequence

# (system_prompt, user_prompt) -> raw model text, or None on failure.
JudgeFn = Callable[[str, str], "str | None"]

DEFAULT_CHUNK_SIZE = 30
DEFAULT_PASSES = 3
DEFAULT_WORKERS = 8


QC_SYSTEM_PROMPT = """You are a NER annotation QC evaluator applying v2 annotation rules.

=== WHAT COUNTS AS A SPAN ===
A span = ONE string element in ONE entity type list.
Example:
  {"entities": {"person": ["Alice", "Bob"], "org": ["Acme Corp"]},
   "json_structures": {"technology": ["fast Fourier transform"]}}
has 4 spans total.

=== YOUR TASK ===
For each row: check every span in output against v2 rules.
Report ONLY clear rule violations — do NOT report correct spans.
- FP: span in output that violates v2 rules (fp_delta=1, fn_delta=0)
- FN: required span missing from output (fp_delta=0, fn_delta=1)
- WRONG_TYPE: span exists but in wrong category (fp_delta=1, fn_delta=1)

CRITICAL: Only report errors you are confident about. Missing spans you are unsure about → skip.

=== OUTPUT ===
Return ONLY valid JSON (no prose, no markdown fences):
{
  "chunk": <int>,
  "total_rows_checked": <int>,
  "failures": [
    {
      "_flat_line_no": <int>,
      "id": "<task_id:row_index>",
      "error_type": "false_positive|missing_entity|wrong_type|missing_json_structure|wrong_bucket|span_boundary",
      "rule_id": "<rule name from v2 rules>",
      "span": "<the specific offending or missing span text>",
      "reason": "<reason in <=30 words>",
      "suggested_action": "<what to add/remove/move>",
      "fp_delta": <0 or 1>,
      "fn_delta": <0 or 1>
    }
  ]
}"""


# ── Text helpers ──────────────────────────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    if "<think>" in text and "</think>" in text:
        text = text[text.rfind("</think>") + len("</think>"):].strip()
    return text.rstrip("\\").strip()


def parse_json_safe(text: str) -> dict | None:
    """Parse model output as JSON, tolerating ``<think>`` blocks and code fences."""
    text = _strip_thinking(text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def count_spans(output: dict | None) -> int:
    """Count annotated spans (string elements across entities + json_structures)."""
    total = 0
    for field_name in ("entities", "json_structures"):
        fval = (output or {}).get(field_name) or {}
        if isinstance(fval, dict):
            for lst in fval.values():
                if isinstance(lst, list):
                    total += sum(1 for s in lst if isinstance(s, str) and s.strip())
    return total


def _qc_user_prompt(rows: Sequence[dict], rules: str, chunk_idx: int) -> str:
    rows_jsonl = "\n".join(
        json.dumps({
            "_flat_line_no": r.get("_flat_line_no", i),
            "id": f"{r.get('task_id', '')}:{r.get('row_index', 0)}",
            "input": r.get("input", ""),
            "output": r.get("output", {}),
        }, ensure_ascii=False)
        for i, r in enumerate(rows)
    )
    return (
        "## V2 Annotation Rules\n"
        f"```\n{rules[:5000]}\n```\n\n"
        "## Task\n"
        f"Evaluate chunk {chunk_idx} — {len(rows)} annotation rows. Report ONLY errors.\n\n"
        "## Rows (JSONL):\n"
        f"{rows_jsonl}"
    )


# ── Single-chunk / single-pass scoring ──────────────────────────────────────────

def evaluate_chunk(
    rows: Sequence[dict],
    rules: str,
    chunk_idx: int,
    judge: JudgeFn,
) -> dict:
    """Score one chunk of rows with a single judge call.

    ``tp`` is derived deterministically (``total_spans - fp``); the judge is
    only asked for the list of violations, never to count correct spans.
    On judge failure the chunk is marked with ``_error`` and excluded from
    aggregation by callers.
    """
    total_spans = sum(count_spans(r.get("output", {})) for r in rows)

    def _err(msg: str) -> dict:
        return {
            "chunk": chunk_idx, "total_rows_checked": len(rows),
            "total_spans": total_spans, "tp": total_spans, "fp": 0, "fn": 0,
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "failures": [], "_error": msg,
        }

    raw = judge(QC_SYSTEM_PROMPT, _qc_user_prompt(rows, rules, chunk_idx))
    if raw is None:
        return _err("judge_call_failed")
    result = parse_json_safe(raw)
    if result is None:
        return {**_err("parse_failed"), "_raw": raw[:500]}

    failures = result.get("failures", []) or []
    fp = sum(int(f.get("fp_delta", 0)) for f in failures)
    fn = sum(int(f.get("fn_delta", 0)) for f in failures)
    tp = max(0, total_spans - fp)
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "chunk": chunk_idx, "total_rows_checked": len(rows),
        "total_spans": total_spans, "tp": tp, "fp": fp, "fn": fn,
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        "failures": failures,
    }


def run_single_pass(
    rows: Sequence[dict],
    rules: str,
    judge: JudgeFn,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    workers: int = DEFAULT_WORKERS,
    verbose: bool = False,
) -> dict:
    """Run the judge once over all rows, chunked and parallelised."""
    chunks = [list(rows[i:i + chunk_size]) for i in range(0, len(rows), chunk_size)]
    chunk_results: list[dict] = [{} for _ in chunks]
    if not chunks:
        return {
            "tp": 0, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 0.0,
            "total_rows": 0, "n_ok_chunks": 0, "n_err_chunks": 0, "failures": [],
        }

    with ThreadPoolExecutor(max_workers=min(workers, len(chunks))) as ex:
        futs = {ex.submit(evaluate_chunk, c, rules, i, judge): i
                for i, c in enumerate(chunks)}
        for fut in as_completed(futs):
            ci = futs[fut]
            chunk_results[ci] = fut.result()

    ok = [c for c in chunk_results if "_error" not in c]
    err = [c for c in chunk_results if "_error" in c]
    if verbose and err:
        print(f"  {len(err)} chunks failed (excluded)", file=sys.stderr)

    tp = sum(c.get("tp", 0) for c in ok)
    fp = sum(c.get("fp", 0) for c in ok)
    fn = sum(c.get("fn", 0) for c in ok)
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        "total_rows": len(rows), "n_ok_chunks": len(ok), "n_err_chunks": len(err),
        "failures": [f for c in ok for f in c.get("failures", [])],
    }


# ── Multi-pass consensus ────────────────────────────────────────────────────────

@dataclass
class ConsensusResult:
    """Majority-vote accuracy estimate across ``passes`` judge runs."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    total_spans: int
    total_rows: int
    passes: int
    threshold: int
    fp_flagged_any: int
    fn_flagged_any: int
    n_err_chunks: int
    per_pass: list[dict] = field(default_factory=list)
    confirmed_failures: list[dict] = field(default_factory=list)

    def meets(self, *, target_precision: float, target_f1: float) -> bool:
        return self.precision >= target_precision and self.f1 >= target_f1

    def to_dict(self) -> dict:
        return {
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "total_spans": self.total_spans, "total_rows": self.total_rows,
            "passes": self.passes, "threshold": self.threshold,
            "fp_flagged_any": self.fp_flagged_any, "fn_flagged_any": self.fn_flagged_any,
            "n_err_chunks": self.n_err_chunks, "per_pass": self.per_pass,
            "confirmed_failures": self.confirmed_failures,
        }


def run_consensus(
    rows: Sequence[dict],
    rules: str,
    judge: JudgeFn,
    *,
    passes: int = DEFAULT_PASSES,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    workers: int = DEFAULT_WORKERS,
    verbose: bool = False,
) -> ConsensusResult:
    """Run the judge ``passes`` times over the SAME rows; keep only errors
    confirmed by a majority of passes.

    FP and FN are scored independently per ``(id, span)`` key. A ``wrong_type``
    failure (``fp_delta=1`` and ``fn_delta=1``) votes in BOTH tallies. The
    denominator is the deterministic span count over the full sample, so a
    chunk that fails on some passes only reduces that span's vote count (a
    conservative bias), never corrupts the totals.
    """
    if passes < 1:
        raise ValueError("passes must be >= 1")
    threshold = passes // 2 + 1  # majority: 2-of-3, 3-of-5, ...
    total_spans = sum(count_spans(r.get("output", {})) for r in rows)

    fp_votes: collections.Counter = collections.Counter()
    fn_votes: collections.Counter = collections.Counter()
    fp_repr: dict[tuple, dict] = {}
    fn_repr: dict[tuple, dict] = {}
    per_pass: list[dict] = []
    err_chunks = 0

    for p in range(passes):
        res = run_single_pass(rows, rules, judge, chunk_size=chunk_size,
                              workers=workers, verbose=verbose)
        per_pass.append({k: res[k] for k in
                         ("precision", "recall", "f1", "fp", "fn", "n_err_chunks")})
        err_chunks += res["n_err_chunks"]
        if verbose:
            print(f"  consensus pass {p + 1}/{passes}: P={res['precision']:.4f} "
                  f"R={res['recall']:.4f} F1={res['f1']:.4f} "
                  f"fp={res['fp']} fn={res['fn']}", file=sys.stderr)
        for f in res.get("failures", []):
            key = (f.get("id"), (f.get("span") or "").strip())
            if not key[1]:
                continue
            if int(f.get("fp_delta", 0)) == 1:
                fp_votes[key] += 1
                fp_repr.setdefault(key, f)
            if int(f.get("fn_delta", 0)) == 1:
                fn_votes[key] += 1
                fn_repr.setdefault(key, f)

    confirmed_fp = [k for k, v in fp_votes.items() if v >= threshold]
    confirmed_fn = [k for k, v in fn_votes.items() if v >= threshold]
    fp = len(confirmed_fp)
    fn = len(confirmed_fn)
    tp = max(0, total_spans - fp)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    confirmed: list[dict] = []
    seen: set[tuple] = set()
    for k in confirmed_fp:
        confirmed.append(fp_repr[k])
        seen.add(k)
    for k in confirmed_fn:
        if k not in seen:
            confirmed.append(fn_repr[k])

    return ConsensusResult(
        precision=round(precision, 4), recall=round(recall, 4), f1=round(f1, 4),
        tp=tp, fp=fp, fn=fn, total_spans=total_spans, total_rows=len(rows),
        passes=passes, threshold=threshold,
        fp_flagged_any=len(fp_votes), fn_flagged_any=len(fn_votes),
        n_err_chunks=err_chunks, per_pass=per_pass, confirmed_failures=confirmed,
    )


# ── Production judge construction ───────────────────────────────────────────────

def build_qc_judge(
    profile,
    *,
    max_tokens: int = 16384,
    timeout: int = 300,
    env=None,
) -> JudgeFn:
    """Build a synchronous judge callable from a resolved :class:`LLMProfile`.

    Uses the OpenAI-compatible Chat Completions API (the ``qc`` target profile
    points at an OpenAI-compatible endpoint). Returns ``None`` on any error so
    the failed chunk is excluded from scoring rather than crashing the run.
    """
    import os

    from openai import OpenAI

    api_key = profile.resolve_api_key(env or os.environ)
    client = OpenAI(api_key=api_key, base_url=profile.base_url)
    model = profile.model
    call_timeout = profile.timeout_seconds or timeout

    def _judge(system: str, user: str) -> str | None:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=max_tokens,
                timeout=call_timeout,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - judge failures are non-fatal
            print(f"  QC judge error ({model}): {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return None

    return _judge


# ── Sampling + orchestration ────────────────────────────────────────────────────

def _source_inputs(task) -> dict:
    out: dict = {}
    for srow in ((getattr(task, "source_ref", None) or {}).get("payload") or {}).get("rows", []):
        ri = srow.get("row_index")
        inp = srow.get("input", "")
        if isinstance(inp, dict):
            inp = inp.get("text", "") or ""
        if ri is not None:
            out[ri] = str(inp) if inp else ""
    return out


def sample_accepted_rows(
    store,
    payload_loader: Callable[[str], "dict | None"],
    *,
    pipeline_id: str,
    sample_count: int,
) -> list[dict]:
    """Load a random sample of accepted-task rows ready for judging.

    ``payload_loader`` returns the latest annotation payload for a task id
    (e.g. ``HumanReviewService._latest_annotation_payload``).
    """
    task_ids = [
        row["task_id"]
        for row in store._conn.execute(
            "SELECT task_id FROM tasks WHERE status='accepted' AND pipeline_id=? "
            "ORDER BY RANDOM() LIMIT ?",
            (pipeline_id, sample_count),
        ).fetchall()
    ]
    rows: list[dict] = []
    for tid in task_ids:
        task = store.load_task(tid)
        if not task:
            continue
        payload = payload_loader(tid)
        if not payload:
            continue
        src = _source_inputs(task)
        for i, row in enumerate(payload.get("rows", [])):
            ri = row.get("row_index", i)
            rows.append({
                "task_id": tid, "row_index": ri, "_flat_line_no": len(rows),
                "input": src.get(ri, ""), "output": row.get("output", {}),
            })
    return rows


def evaluate_accuracy(
    store,
    payload_loader: Callable[[str], "dict | None"],
    rules: str,
    judge: JudgeFn,
    *,
    pipeline_id: str,
    sample_count: int,
    passes: int = DEFAULT_PASSES,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    workers: int = DEFAULT_WORKERS,
    verbose: bool = False,
) -> ConsensusResult:
    """Sample accepted tasks and return the consensus accuracy estimate."""
    rows = sample_accepted_rows(
        store, payload_loader, pipeline_id=pipeline_id, sample_count=sample_count,
    )
    return run_consensus(
        rows, rules, judge,
        passes=passes, chunk_size=chunk_size, workers=workers, verbose=verbose,
    )
