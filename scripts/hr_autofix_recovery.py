"""
HR auto-fix recovery: for tasks routed to HR for mechanical_fail / verbatim-
exhausted-style reasons, run auto_fix_safe_spans_in_place on the best
available payload (arbiter's corrected_annotation if any, else current
annotation_result). If after fix all four validators pass (schema, verbatim,
cross-type, trailing-punct), persist the cleaned payload as a fresh
annotation_result artifact and transition HR → ACCEPTED.

Usage:
    python hr_autofix_apply.py             # dry-run
    python hr_autofix_apply.py --apply     # actually act
"""
import argparse, json, re, sqlite3, pathlib, sys
from annotation_pipeline_skill.store.sqlite_store import SqliteStore
from annotation_pipeline_skill.core.models import TaskStatus, ArtifactRef
from annotation_pipeline_skill.core.transitions import transition_task, InvalidTransition
from annotation_pipeline_skill.core.schema_validation import (
    auto_fix_safe_spans_in_place, find_verbatim_violations,
    find_cross_type_collisions, find_trailing_punctuation_spans,
    validate_payload_against_task_schema, SchemaValidationError,
)
from annotation_pipeline_skill.runtime.subagent_cycle import _parse_llm_json

ap = argparse.ArgumentParser()
ap.add_argument('--apply', action='store_true')
ap.add_argument('--project-root', default='projects/v3_initial_deployment')
args = ap.parse_args()

root = pathlib.Path(args.project_root) / '.annotation-pipeline'
store = SqliteStore(root=root, db_path=root/'db.sqlite')
conn = store._conn; conn.row_factory = sqlite3.Row

AUTOFIXABLE_MARKERS = (
    'Arbiter retried', 'mechanical retries',
    'could not produce a verbatim-compliant', 'Arbiter ruled qc/neither but could not',
    'Second arbiter tried to correct',
)

def latest_arbiter_payload(task_id):
    arts = [a for a in store.list_artifacts(task_id) if a.kind == 'arbiter_result']
    if not arts: return None, None
    art = arts[-1]
    p = root / art.path
    if not p.exists(): return None, None
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None, None
    ca = (d.get('decision') or {}).get('corrected_annotation')
    return (ca if isinstance(ca, dict) else None), art.path

def latest_annotation_payload(task_id):
    arts = [a for a in store.list_artifacts(task_id) if a.kind == 'annotation_result']
    if not arts: return None, None
    art = arts[-1]
    p = root / art.path
    if not p.exists(): return None, None
    try:
        outer = json.loads(p.read_text())
        text = outer.get('text') if isinstance(outer, dict) else None
        return (_parse_llm_json(text) if isinstance(text, str) else None), art.path
    except Exception:
        return None, None

def next_attempt_id(task):
    suffixes = []
    for a in store.list_attempts(task.task_id):
        m = re.search(r"-attempt-(\d+)(?:-|$)", a.attempt_id)
        if m: suffixes.append(int(m.group(1)))
    return f"{task.task_id}-attempt-{max(suffixes, default=-1) + 1}"

rows = conn.execute("""
  SELECT t.task_id,
    (SELECT reason FROM audit_events e WHERE e.task_id=t.task_id AND e.next_status='human_review' ORDER BY e.rowid DESC LIMIT 1) reason
  FROM tasks t WHERE t.pipeline_id='v3_initial_deployment' AND t.status='human_review'
""").fetchall()

candidates = []
for r in rows:
    reason = r['reason'] or ''
    if any(m in reason for m in AUTOFIXABLE_MARKERS):
        candidates.append(r['task_id'])

would, skipped_dirty, skipped_nopay = 0, 0, 0
applied = []
for tid in candidates:
    t = store.load_task(tid)
    if t.status is not TaskStatus.HUMAN_REVIEW:
        continue
    payload, src_path = latest_arbiter_payload(tid)
    source_tag = 'arbiter_correction'
    if payload is None:
        payload, src_path = latest_annotation_payload(tid)
        source_tag = 'annotator'
    if payload is None:
        skipped_nopay += 1; continue
    n_fix = auto_fix_safe_spans_in_place(t, payload)
    try:
        validate_payload_against_task_schema(t, payload, store=store)
        ok_schema = True
    except SchemaValidationError:
        ok_schema = False
    vbio = find_verbatim_violations(t, payload)
    coll = find_cross_type_collisions(payload)
    tpun = find_trailing_punctuation_spans(t, payload)
    if ok_schema and not vbio and not coll and not tpun:
        would += 1
        if not args.apply:
            continue
        # Apply: write fresh annotation_result + transition. entity_statistics
        # is recount-only now (rebuilt on demand via Re-count / recount_project),
        # so this recovery path no longer bumps stats inline.
        attempt_id = next_attempt_id(t)
        t.current_attempt += 1
        rel_path = f"artifact_payloads/{t.task_id}/{attempt_id}_hr_autofix_recovery.json"
        out_path = store.root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_text = json.dumps(payload, sort_keys=True, indent=2)
        out_path.write_text(json.dumps({
            'text': cleaned_text,
            'task_id': t.task_id,
            'source': f'hr_autofix_recovery({source_tag})',
            'diagnostics': {'resolution_source': 'hr_autofix_recovery',
                            'source_artifact': src_path,
                            'autofix_rewrites': n_fix},
        }, indent=2) + '\n', encoding='utf-8')
        artifact = ArtifactRef.new(
            task_id=t.task_id, kind='annotation_result', path=rel_path,
            content_type='application/json',
            metadata={'source': 'hr_autofix_recovery', 'attempt_id': attempt_id,
                      'source_artifact_kind': source_tag,
                      'autofix_rewrites': n_fix},
        )
        store.append_artifact(artifact)
        try:
            ev = transition_task(
                t, TaskStatus.ACCEPTED,
                actor='operator', stage='hr_autofix_recovery',
                reason=f'HR auto-fix recovery: auto_fixed={n_fix} spans, all validators pass after new auto-fix logic',
                attempt_id=attempt_id,
                metadata={'recovery': 'hr_autofix_recovery',
                          'autofix_rewrites': n_fix,
                          'source_artifact_kind': source_tag},
            )
            store.save_task(t)
            store.append_event(ev)
            applied.append(tid)
        except InvalidTransition as exc:
            print(f'  skip {tid}: {exc}'); skipped_dirty += 1
    else:
        skipped_dirty += 1

print(f'mode: {"APPLY" if args.apply else "DRY-RUN"}')
print(f'candidates: {len(candidates)}')
print(f'would_accept (or actually accepted): {would}')
print(f'still_dirty (real issue):            {skipped_dirty}')
print(f'no_payload:                          {skipped_nopay}')
if args.apply:
    print(f'\napplied {len(applied)} tasks')
