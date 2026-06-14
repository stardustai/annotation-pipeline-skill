# Consensus (N-way) Annotation

Set `stages.annotation.replicas: N` in a project's `workflow.yaml` to run N
independent annotators per task. Spans agreed by `keep_threshold` of them are
kept; the rest go to `arbiter_target`, which resolves conflicts (选择题) and fills
clear gaps (补漏). `replicas: 1` (the default) is the legacy single-annotator flow.

## Config

```yaml
stages:
  annotation:
    replicas: 2
    targets: [qwen3.6-35b-a3b, MiniMax-M3]   # one target per replica
    keep_threshold: 2                        # 2 = keep only unanimous spans
    on_disagree: arbiter                     # arbiter | drop
    arbiter_target: MiniMax-M3
    accept_directly: true                    # dual+arbiter IS the gate — no QC stage
```

- `replicas` — number of annotators run per task.
- `targets` — exactly `replicas` profile names (or a single name, broadcast to N). Use **different** models for the diversity that makes agreement meaningful.
- `keep_threshold` — a span is auto-kept if it appears in ≥ this many drafts. `replicas` = unanimous (recommended for N=2), `1` = union.
- `on_disagree` — `arbiter` resolves below-threshold spans + adds clear misses; `drop` discards them.
- `arbiter_target` — the model that arbitrates. Needs to be **reliable** (valid output every task), not necessarily frontier.
- `accept_directly` — `true` makes the task ACCEPT straight after the arbiter merge, with **no separate QC stage** (the arbiter is the quality gate).

## Why

Empirically (v5_ner_phrase, 24 tasks, gate口径): every dual+arbiter configuration
scored F1 0.97–0.99 regardless of annotator/arbiter strength (qwen+Haiku+qwen-arbiter
= 0.971; qwen+M3 = 0.994), vs single annotators 0.93–0.98. The **structure** carries
the quality — two drafts give a high-recall union, and selecting among candidates is
easy even for weak models. Model strength buys *reliability* (a weak arbiter dropped
25% of tasks on malformed output) and the last ~1 F1 point. Sweet spot: **N=2,
keep_threshold=2**, with a reliable arbiter. N>2 mostly adds cost.
