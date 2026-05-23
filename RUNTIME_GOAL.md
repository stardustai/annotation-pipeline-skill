# Runtime Health Goals

流水线运行健康度的量化目标。每个指标给出当前基线、近期目标和理想目标，以及测量 SQL。

---

## 当前基线快照（v3_initial_deployment，2026-05-23）

| 指标 | 当前值 | 近期目标 | 理想目标 |
|------|--------|----------|----------|
| 驳回率（退回重标注次数/任务数）| 2.61²  | < 1.0 | < 0.5 |
| HR 升级率（全部）| 18.8% | < 5% | < 1% |
| — 其中 arbiter 主动标记不确定 | 16.5% | < 3% | < 0.5% |
| — 其中 arbiter 重试耗尽（3 次）| 1.3% | < 0.5% | 0% |
| Accepted 任务平均 LLM 调用次数 | 15.2 次 | < 10 次 | < 6 次 |
| 进入 arbitration 的任务比例 | 83% | < 50% | < 20% |
| QC JSON 解析失败率 | 1.7% | < 0.5% | < 0.1% |
| Annotation avg 耗时 | 71s | < 60s p50 | < 45s p50 |
| QC avg 耗时 | 87s | < 45s p50 | < 30s p50 |
| Arbitration avg 耗时 | 75s | < 60s p50 | < 45s p50 |

> ² 有效 QC/arbiter 退回（subagent-runtime 发起）：10850 次 ÷ 4151 任务 = 2.61。另有 2174 次 scheduler recovery 退回不计入。

---

## 1. 结果质量（Outcome Quality）

### 1.1 驳回率
**定义**：QC 或 arbiter 将任务退回重新标注的次数 ÷ 总任务数。即每个任务平均被退回几次。

```
驳回率 = #back_to_annotation_events / #tasks
```

退回事件 = `audit_events` 中 `previous_status IN ('qc','arbitrating')` 且 `next_status IN ('pending','annotating')` 且 `actor = 'subagent-runtime'`（排除 scheduler recovery 和手动操作）。

```
🟢 < 0.5    🟡 0.5–1.5   🔴 > 1.5
（当前基线：2.61 次/任务）
```

**解读**：< 0.5 意味着大多数任务首轮 QC 通过（少量重试）；> 1.5 意味着平均每任务反复标注 2+ 轮，是效率损耗的主要来源。

```sql
-- 驳回率（有效 QC/arbiter 退回，排除 scheduler recovery）
SELECT
  COUNT(*) * 1.0 / (SELECT COUNT(*) FROM tasks) AS rejection_rate,
  COUNT(*) AS total_back_events,
  (SELECT COUNT(*) FROM tasks) AS total_tasks
FROM audit_events
WHERE previous_status IN ('qc', 'arbitrating')
  AND next_status IN ('pending', 'annotating')
  AND actor = 'subagent-runtime';
```

```sql
-- 按退回次数分布（了解尾部）
SELECT back_count, COUNT(*) AS tasks,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
FROM (
  SELECT task_id, COUNT(*) AS back_count
  FROM audit_events
  WHERE previous_status IN ('qc','arbitrating')
    AND next_status IN ('pending','annotating')
    AND actor = 'subagent-runtime'
  GROUP BY task_id
)
GROUP BY back_count ORDER BY back_count;
```

---

### 1.2 HR 升级率（全部及分类）
**定义**：当前处于 `human_review` 状态的任务占全部终态任务的比例，以及各原因的细分。

```
总 HR 率:     🟢 < 3%      🟡 3–10%     🔴 > 10%
Arbiter 不确定:  🟢 < 1%   🟡 1–5%      🔴 > 5%
Arbiter 重试耗尽: 🟢 0%    🟡 < 0.5%    🔴 > 0.5%
```

当前 18.8% 的 HR 率中，**88%（511 条）是 arbiter 主动标记 tentative/unsure**，这是最大的改进空间。

```sql
-- HR 分类（按每个任务最近一次进入 HR 的原因）
WITH latest_hr AS (
  SELECT ae.task_id, ae.reason
  FROM audit_events ae
  WHERE ae.next_status = 'human_review'
    AND ae.task_id IN (SELECT task_id FROM tasks WHERE status = 'human_review')
    AND ae.seq = (
      SELECT MAX(ae2.seq) FROM audit_events ae2
      WHERE ae2.task_id = ae.task_id AND ae2.next_status = 'human_review'
    )
),
total AS (SELECT count(*) AS n FROM tasks WHERE status IN ('accepted','rejected','human_review'))
SELECT
  (SELECT count(*) FROM tasks WHERE status='human_review') * 100.0 / total.n AS hr_total_pct,
  (SELECT count(*) FROM latest_hr WHERE reason LIKE 'Arbiter flagged%') * 100.0 / total.n AS hr_uncertain_pct,
  (SELECT count(*) FROM latest_hr WHERE reason LIKE 'Arbiter retried%') * 100.0 / total.n AS hr_stuck_pct,
  (SELECT count(*) FROM latest_hr WHERE reason LIKE '%worker bailed%') * 100.0 / total.n AS hr_bail_pct,
  (SELECT count(*) FROM latest_hr WHERE reason LIKE '%zombie%' OR reason LIKE '%auto-escalated%') * 100.0 / total.n AS hr_system_pct
FROM total;
```

---

## 2. 效率（LLM Run Efficiency）

### 2.1 Accepted 任务平均 LLM 调用次数
**定义**：每个最终 `accepted` 的任务，在 annotation + qc + arbitration 三个 stage 累计的 attempts 总数（不含 prelabel）。

```
🟢 < 6      🟡 6–10     🔴 > 10
```

**理解**：

| 路径 | 调用次数 |
|------|---------|
| 首轮通过（1 ann + 1 QC）| 2 |
| 一次修改（2 ann + 2 QC）| 4 |
| 两次修改（3 ann + 3 QC）| 6 |
| 两次修改 + arbitration | 7 |

当前平均 15.2 意味着大多数任务经历了 **6–7 轮 annotation-QC 循环**，是最高优先级的效率问题。

```sql
SELECT
  AVG(cnt) AS avg_runs,
  MIN(cnt), MAX(cnt),
  SUM(CASE WHEN cnt <= 2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pct_le2,
  SUM(CASE WHEN cnt <= 4 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pct_le4,
  SUM(CASE WHEN cnt <= 6 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pct_le6
FROM (
  SELECT task_id, COUNT(*) AS cnt FROM attempts
  WHERE stage IN ('annotation','qc','arbitration')
    AND provider_id != 'prelabel'
    AND task_id IN (SELECT task_id FROM tasks WHERE status = 'accepted')
  GROUP BY task_id
);
```

---

### 2.2 首轮通过率
**定义**：accepted 任务中，仅经过 1 annotation + 1 QC（attempts ≤ 2）即通过的比例。

```
🟢 > 30%    🟡 10–30%   🔴 < 10%
```

**当前：0%**（最低 4 次调用，说明即使最顺利的任务也经历了至少 2 轮循环）。这直接反映了 annotator 首次输出质量与 QC 标准的匹配度。

---

### 2.3 Arbitration 进入率
**定义**：accepted 任务中，有过至少 1 次 arbitration attempt 的比例。

```
🟢 < 20%    🟡 20–50%   🔴 > 50%
```

**当前：83%**。Arbitration 每次平均 75s，当前每任务平均 2.5 次 = 占总 LLM 时间约 19%。降低 arbitration 进入率和重试次数，是减少总耗时的关键。

```sql
SELECT
  SUM(CASE WHEN arb_cnt > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS arb_rate
FROM (
  SELECT t.task_id,
    SUM(CASE WHEN a.stage = 'arbitration' THEN 1 ELSE 0 END) AS arb_cnt
  FROM tasks t LEFT JOIN attempts a ON t.task_id = a.task_id
  WHERE t.status = 'accepted'
  GROUP BY t.task_id
);
```

---

### 2.4 Arbiter Verbatim Bail 率
**定义**：accepted 任务中，arbiter 因无法生成 verbatim-compliant 修订而 bail 的任务比例（bail_count > 0 / bail_count ≥ 3）。

Bail ≥ 3 次会直接触发 HR 升级，是当前 1.3% HR-stuck 的直接原因。

```
🟢 bail>0: < 5%   bail≥3: 0%
🟡 bail>0: 5–15%  bail≥3: < 0.5%
🔴 bail>0: > 15%  bail≥3: > 0.5%
```

```sql
SELECT
  SUM(CASE WHEN CAST(json_extract(metadata_json,'$.arbiter_verbatim_bail_count') AS INT) > 0
      THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS bail_any_pct,
  SUM(CASE WHEN CAST(json_extract(metadata_json,'$.arbiter_verbatim_bail_count') AS INT) >= 3
      THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS bail_ge3_pct
FROM tasks WHERE status = 'accepted';
```

---

### 2.5 QC 解析失败率
**定义**：QC stage 中 `status='failed'` 且 `error_json.kind='parse_error'` 的比例。

```
🟢 < 0.5%   🟡 0.5–2%   🔴 > 2%
```

**当前：1.7%**。每次 QC 解析失败等于浪费一次 QC 调用，并强制 annotation 再循环一轮（+2 次调用）。降到 0.5% 以下预计可减少 ~5% 的总调用量。

```sql
SELECT
  SUM(CASE WHEN status = 'failed'
        AND json_extract(error_json,'$.kind') = 'parse_error'
      THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS qc_parse_fail_pct,
  COUNT(*) AS total_qc_attempts
FROM attempts WHERE stage = 'qc';
```

---

## 3. 性能（LLM Call Performance）

### 3.1 各 Stage 平均耗时

| Stage | p50 目标 | p95 目标 | 超时率（≥ 900s）目标 |
|-------|----------|----------|---------------------|
| annotation | < 60s | < 180s | < 0.5% |
| qc | < 45s | < 120s | < 0.3% |
| arbitration | < 60s | < 180s | < 0.5% |

**当前**：annotation avg 71s，qc avg 87s，arbitration avg 75s。三者均有超时记录（max=900s）。qc 平均耗时高于 annotation 不合理，说明 QC prompt 或 provider 有问题。

```sql
SELECT stage,
  ROUND(AVG((julianday(finished_at)-julianday(started_at))*86400), 1) AS avg_s,
  COUNT(*) AS n,
  SUM(CASE WHEN (julianday(finished_at)-julianday(started_at))*86400 >= 890
      THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS timeout_pct
FROM attempts
WHERE status = 'succeeded'
  AND provider_id NOT IN ('prelabel')
  AND finished_at IS NOT NULL AND started_at IS NOT NULL
GROUP BY stage;
```

---

### 3.2 Worker Bail 率（每任务）
**定义**：accepted / rejected 任务中，`worker_bail_count > 0` 的比例，以及 ≥ 5 次（触发自动 HR 升级门槛）的比例。

```
🟢 bail>0: < 3%    bail≥5: 0%
🟡 bail>0: 3–10%   bail≥5: < 0.5%
🔴 bail>0: > 10%   bail≥5: > 0.5%
```

```sql
SELECT
  SUM(CASE WHEN CAST(json_extract(metadata_json,'$.worker_bail_count') AS INT) > 0
      THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS bail_any_pct,
  SUM(CASE WHEN CAST(json_extract(metadata_json,'$.worker_bail_count') AS INT) >= 5
      THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS bail_ge5_pct
FROM tasks WHERE status IN ('accepted','rejected');
```

---

### 3.3 vLLM Prefix Cache 命中率（qwen 系列）
**定义**：`provider_id LIKE 'qwen%'` 调用中，`cache_read_input_tokens / input_tokens`。

```
🟢 > 30%    🟡 10–30%   🔴 < 10%
```

这是本次引入 `openai_sdk` runtime 的核心动机（消除 claude CLI billing header 破坏前缀 cache 的问题）。命中率 < 10% 说明 byte-stable 前缀未生效，优先检查 `system` prompt 是否有逐次变化的字段。

**测量**：当前 attempts 表未存储 token breakdown，需从 `LLMGenerateResult.usage` 中取 `cache_read_input_tokens`，建议后续在 attempts.artifacts_json 中记录。

---

## 4. 系统吞吐（Throughput）

### 4.1 每小时完成任务数
```
🟢 > 20 tasks/h   🟡 8–20 tasks/h   🔴 < 8 tasks/h
```

（理论上限：8 个并发 worker × 每任务 ~15min = ~32/h；受 LLM 平均耗时约束。）

```sql
SELECT COUNT(*) AS completed_last_hour
FROM audit_events
WHERE next_status IN ('accepted','rejected')
  AND created_at >= datetime('now','-1 hour');
```

---

### 4.2 Pending 队列消耗趋势
**定义**：`pending` 状态任务数应随时间下降（系统消化速度 > 新增速度）。

```
🟢 每小时净减少 > 10   🟡 持平（±5%）   🔴 持续增长
```

```sql
SELECT COUNT(*) AS pending_now FROM tasks WHERE status = 'pending';
-- 对比 1h 前快照判断方向
```

---

## 5. 综合健康检查命令

在项目根目录运行，输出所有关键指标快照：

```bash
python3 - <<'EOF'
import sqlite3

DB = "projects/v3_initial_deployment/.annotation-pipeline/db.sqlite"
conn = sqlite3.connect(DB)

def q(sql, *args):
    return conn.execute(sql, args).fetchone()[0]

# --- Task distribution ---
rows = conn.execute(
    "SELECT status, count(*) FROM tasks GROUP BY status ORDER BY count(*) DESC"
).fetchall()
print("=== Task Distribution ===")
for s, n in rows:
    print(f"  {s:20s}  {n:6d}")

terminal = q("SELECT count(*) FROM tasks WHERE status IN ('accepted','rejected','human_review')")
accepted = q("SELECT count(*) FROM tasks WHERE status='accepted'")
hr_count = q("SELECT count(*) FROM tasks WHERE status='human_review'")

print(f"\n=== Key Metrics (terminal={terminal}) ===")

# Rejection rate (back to annotation)
back_events = q("""
  SELECT COUNT(*) FROM audit_events
  WHERE previous_status IN ('qc','arbitrating')
    AND next_status IN ('pending','annotating')
    AND actor = 'subagent-runtime'
""")
total_tasks = q("SELECT COUNT(*) FROM tasks")
print(f"  Rejection rate (back/task): {back_events/total_tasks:.2f}    target < 0.5")

# HR rates
hr_uncertain = q("""
  SELECT count(DISTINCT ae.task_id) FROM audit_events ae
  WHERE ae.next_status='human_review'
    AND ae.task_id IN (SELECT task_id FROM tasks WHERE status='human_review')
    AND ae.reason LIKE 'Arbiter flagged%'
    AND ae.seq=(SELECT MAX(ae2.seq) FROM audit_events ae2
                WHERE ae2.task_id=ae.task_id AND ae2.next_status='human_review')
""")
hr_stuck = q("""
  SELECT count(DISTINCT ae.task_id) FROM audit_events ae
  WHERE ae.next_status='human_review'
    AND ae.task_id IN (SELECT task_id FROM tasks WHERE status='human_review')
    AND ae.reason LIKE 'Arbiter retried%'
    AND ae.seq=(SELECT MAX(ae2.seq) FROM audit_events ae2
                WHERE ae2.task_id=ae.task_id AND ae2.next_status='human_review')
""")
print(f"  HR rate (total):           {hr_count/terminal*100:.1f}%  target < 3%")
print(f"  HR rate (arbiter unsure):  {hr_uncertain/terminal*100:.1f}%  target < 1%")
print(f"  HR rate (arbiter stuck):   {hr_stuck/terminal*100:.1f}%  target < 0.5%")

# Avg LLM runs
avg_runs = q("""
  SELECT AVG(cnt) FROM (
    SELECT task_id, COUNT(*) AS cnt FROM attempts
    WHERE stage IN ('annotation','qc','arbitration') AND provider_id!='prelabel'
      AND task_id IN (SELECT task_id FROM tasks WHERE status='accepted')
    GROUP BY task_id)
""")
print(f"  Avg LLM runs (accepted):   {avg_runs:.1f}    target < 6")

# Arbitration rate
arb_rate = q("""
  SELECT SUM(has_arb)*100.0/COUNT(*) FROM (
    SELECT t.task_id, MAX(CASE WHEN a.stage='arbitration' THEN 1 ELSE 0 END) AS has_arb
    FROM tasks t LEFT JOIN attempts a ON t.task_id=a.task_id
    WHERE t.status='accepted' GROUP BY t.task_id)
""")
print(f"  Arbitration rate:          {arb_rate:.0f}%   target < 20%")

# QC parse fail
qc_fail = q("""
  SELECT SUM(CASE WHEN status='failed'
    AND json_extract(error_json,'$.kind')='parse_error' THEN 1 ELSE 0 END)*100.0/count(*)
  FROM attempts WHERE stage='qc'
""")
print(f"  QC parse fail rate:        {qc_fail:.2f}%  target < 0.5%")

# Stage latency
print("\n=== Stage Latency (avg seconds, succeeded) ===")
for row in conn.execute("""
    SELECT stage,
      round(avg((julianday(finished_at)-julianday(started_at))*86400),1) AS avg_s,
      sum(CASE WHEN (julianday(finished_at)-julianday(started_at))*86400 >= 890
          THEN 1 ELSE 0 END)*100.0/count(*) AS timeout_pct,
      count(*) AS n
    FROM attempts WHERE status='succeeded' AND provider_id!='prelabel'
      AND finished_at IS NOT NULL AND started_at IS NOT NULL
    GROUP BY stage
""").fetchall():
    print(f"  {row[0]:12s}  avg={row[1]}s  timeout={row[2]:.2f}%  n={row[3]}")

# Pending
pending = q("SELECT count(*) FROM tasks WHERE status='pending'")
print(f"\n=== Queue ===")
print(f"  Pending:  {pending}")

conn.close()
EOF
```

---

## 6. 指标演进路径

| 阶段 | 主要工作 | 驳回率（次/任务）| Avg LLM runs | Arb rate | HR rate |
|------|----------|-----------------|--------------|----------|---------|
| **Phase 0（当前）** | — | 2.61 | 15.2 | 83% | 18.8% |
| **Phase 1** | 消灭 QC parse error；arbiter uncertain 减少 50% | < 2.0 | < 12 | < 70% | < 10% |
| **Phase 2** | Annotator 首轮通过率提升（prompt + few-shot）；arbitration early-accept | < 1.0 | < 8 | < 40% | < 4% |
| **Phase 3** | Prefix cache hit > 30%；arbitration 进入门槛收紧 | < 0.5 | < 6 | < 20% | < 2% |
| **Phase 4（理想）** | 首轮通过率 > 30%；bail 率 < 1% | < 0.3 | < 4 | < 10% | < 1% |

---

*最后更新：2026-05-23*  
*数据来源：`projects/v3_initial_deployment/.annotation-pipeline/db.sqlite`*
