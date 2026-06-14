# Consensus (N-way) Annotation

Set `stages.annotation.replicas: N` in a project's `workflow.yaml` to run N
independent annotators per task. Spans agreed by `keep_threshold` of them are
kept; the rest go to `arbiter_target`, which resolves conflicts (选择题) and fills
clear gaps (补漏). `replicas: 1` (the default) is the legacy single-annotator flow.

## Config

Define one target per annotator in `llm_profiles.yaml`, then list them in `workflow.yaml`:

```yaml
# llm_profiles.yaml  (targets map: name -> model profile)
targets:
  annotation: MiniMax-M3        # annotator A
  annotation_2: claude_sonnet   # annotator B  (add a dedicated second-annotator target)
  arbiter: codex_5.5            # arbitration model
  qc: glm_51                    # QC model (idle in multi-annotation)
```
```yaml
# workflow.yaml
stages:
  annotation:
    replicas: 2
    targets: [annotation, annotation_2]   # one purpose-named annotator target per replica
    keep_threshold: 2                     # 2 = keep only unanimous spans; the rest go to the arbiter
    on_disagree: arbiter                  # arbiter | drop
    arbiter_target: arbiter               # the real arbitration target
    # accept_directly: omitted → defaults true for replicas>1 (QC is disabled in multi-annotation)
```

- `replicas` — number of annotators run per task.
- `targets` — exactly `replicas` **registry target names** (keys under `targets:` in `llm_profiles.yaml`), **not** raw profile names. Give each annotator its **own** target (e.g. `annotation`, `annotation_2`) pointing at a **different** model — don't reuse the `qc`/`arbiter` targets as annotators; that conflates roles. Diversity between the two models is what makes agreement meaningful.
- `keep_threshold` — a span is auto-kept if it appears in ≥ this many drafts. `replicas` = unanimous (recommended for N=2), `1` = union.
- `on_disagree` — `arbiter` resolves below-threshold spans + adds clear misses; `drop` discards them.
- `arbiter_target` — the **arbitration** target that reconciles the N drafts (default `arbiter`). Must be **reliable** (valid output every task). QC is a single validator and cannot arbitrate, so don't point this at `qc`; use a real arbiter model.
- `accept_directly` — ACCEPT straight after the arbiter merge with **no QC stage**. **Defaults to `true` when `replicas > 1`** — in multi-annotation the arbiter *is* the quality gate, so QC (which only validates a single annotation) is disabled. Set `false` to additionally run QC after the merge. `replicas == 1` keeps the normal single-annotator + QC flow.

## Why

Empirically (v5_ner_phrase, 24 tasks, gate口径): every dual+arbiter configuration
scored F1 0.97–0.99 regardless of annotator/arbiter strength (qwen+Haiku+qwen-arbiter
= 0.971; qwen+M3 = 0.994), vs single annotators 0.93–0.98. The **structure** carries
the quality — two drafts give a high-recall union, and selecting among candidates is
easy even for weak models. Model strength buys *reliability* (a weak arbiter dropped
25% of tasks on malformed output) and the last ~1 F1 point. Sweet spot: **N=2,
keep_threshold=2**, with a reliable arbiter. N>2 mostly adds cost.
