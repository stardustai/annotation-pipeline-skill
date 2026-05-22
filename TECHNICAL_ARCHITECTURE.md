# Annotation Pipeline Skill 技术架构文档

## 1. 文档目标

本文档定义 `annotation-pipeline-skill` 的目标技术架构。重点不是复刻 `memory-ner` 的脚本实现，而是把其中通用、可复用的工程机制抽象为开源 skill 的内核。


## 2. 设计目标

### 2.1 主要目标

- 支持任意标注任务类型，不写死 NER
- 支持多阶段流水线，而不是单步脚本
- 保证 task 级 traceability
- 支持 deterministic gate、QC、repair、merge
- 支持本地最小运行模式和可替换的重型运行模式
- 让业务特定逻辑以插件形式接入

### 2.2 非目标

- 不把所有运行模式都塞进核心
- 不要求默认依赖 Redis、Docker、systemd
- 不让 provider、schema、dataset 逻辑污染框架核心


## 3. 总体架构

建议采用分层架构：

1. `Core Domain`
2. `Application Services`
3. `Plugin Contracts`
4. `Runtime Backends`
5. `Interfaces`
6. `Integration Adapters`

### 3.1 分层说明

#### Core Domain

负责：

- task 状态机
- attempt 模型
- artifact 元数据
- transition rules
- audit event 结构

不负责：

- 如何调用模型
- 如何读取特定数据源
- 如何落地到 Redis / systemd / Web UI

#### Application Services

负责：

- 创建 task
- 推进阶段
- 执行 retry policy
- 组织 validate / qc / merge 调用
- 协调 store、runtime、plugins

#### Plugin Contracts

负责定义接口：

- dataset adapter
- validator
- prompt builder
- qc policy
- repair strategy
- merge sink
- provider client

#### Runtime Backends

可选实现：

- local subprocess
- queue + worker
- systemd-based runtime

#### Interfaces

用户接入层：

- CLI
- dashboard API
- TypeScript Web UI

#### Integration Adapters

负责外部系统边界：

- 外部任务 API 拉取
- 外部 task id 与内部 task id 映射
- 阶段状态回传
- 结果提交
- 幂等、重试和 dead-letter 记录

Integration adapter 不能直接改写 task JSON，必须通过 application service 创建 task 和推进状态。


## 4. 核心模块划分

建议目录：

```text
annotation_pipeline_skill/
  core/
    models.py
    states.py
    transitions.py
    events.py
  services/
    task_factory.py
    pipeline_service.py
    retry_service.py
    merge_service.py
    dashboard_service.py
    settings_service.py
    external_task_service.py
    feedback_service.py
  store/
    base.py
    file_store.py
  runtime/
    base.py
    local_subprocess.py
    queued_runtime.py
  plugins/
    base.py
    registry.py
  interfaces/
    cli.py
    api.py
  web/
    package.json
    tsconfig.json
    src/
      api/
      components/
      pages/
      types/
  templates/
    project/
    adapters/
  examples/
    jsonl_demo/
```


## 5. 核心领域模型

### 5.1 ProjectConfig

```python
ProjectConfig:
  project_id: str
  root_dir: str
  task_store_backend: str
  runtime_backend: str
  concurrency: dict
  plugins: dict
  providers: dict
  annotators: dict
  stage_routes: dict
  external_task_api: dict | None
```

### 5.2 Task

```python
Task:
  task_id: str
  pipeline_id: str
  source_ref: SourceRef
  external_ref: ExternalTaskRef | None
  modality: str
  annotation_requirements: dict
  selected_annotator_id: str | None
  status: TaskStatus
  current_attempt: int
  assignee: str | None
  created_at: datetime
  updated_at: datetime
  active_run_id: str | None
  next_retry_at: datetime | None
  metadata: dict
```

### 5.2.1 ExternalTaskRef

```python
ExternalTaskRef:
  system_id: str
  external_task_id: str
  source_url: str | None
  idempotency_key: str
  last_status_posted: str | None
  last_status_posted_at: datetime | None
  submit_attempts: int
```

`ExternalTaskRef` 是 integration 边界元数据。core 可以保存引用，但不能包含外部 API client 逻辑。

### 5.3 Attempt

```python
Attempt:
  attempt_id: str
  task_id: str
  index: int
  stage: StageName
  status: AttemptStatus
  started_at: datetime | None
  finished_at: datetime | None
  provider_id: str | None
  model: str | None
  effort: str | None
  route_role: str | None  # primary | fallback | override
  summary: str | None
  error: ErrorInfo | None
  artifacts: list[ArtifactRef]
```

### 5.3.1 AnnotatorProfile

```python
AnnotatorProfile:
  annotator_id: str
  display_name: str
  modality: list[str]  # text | image | video | point_cloud
  annotation_types: list[str]
  input_artifact_kinds: list[str]
  output_artifact_kinds: list[str]
  provider_route_id: str | None
  external_tool_id: str | None
  preview_renderer_id: str | None
  human_review_policy_id: str | None
  fallback_annotator_id: str | None
  enabled: bool
  metadata: dict
```

`AnnotatorProfile` 描述“谁能标什么”。它和 `ProviderConfig` 分离：provider 解决调用模型或服务的问题，annotator profile 解决 task 能力匹配、输出 artifact，以及是否为 QC/Human Review 生成 preview 证据的问题。

### 5.4 ArtifactRef

```python
ArtifactRef:
  artifact_id: str
  task_id: str
  kind: str
  path: str
  content_type: str
  created_at: datetime
  metadata: dict
```

多模态 artifact 应通过 `kind` 和 `metadata` 描述输入输出。例如：

- `image_source`：原图路径、尺寸、颜色空间
- `image_bbox_annotation`：box 坐标、label、confidence、source model
- `image_bbox_preview`：渲染后的 overlay 图片
- `video_frame_annotation`：frame index、timestamp、box/mask/track
- `point_cloud_annotation`：3D box、coordinate frame、instance id
- `human_review_answer`：人工复核阶段写入的最终答案，schema 校验通过后由 export 服务优先于 `annotation_result` 选用

### 5.5 FeedbackRecord

```python
FeedbackRecord:
  feedback_id: str
  task_id: str
  attempt_id: str | None
  source: str  # validator | qc_provider | human_review | merge_gate
  severity: str  # info | warning | error | blocker
  code: str
  message: str
  location: FeedbackLocation | None
  artifact_refs: list[ArtifactRef]
  suggested_action: str  # bulk_code_repair | annotator_rerun | manual_annotation | reject
  repair_decision: str | None
  status: str  # open | applied | dismissed | superseded
  created_at: datetime
  resolved_at: datetime | None
  metadata: dict

FeedbackLocation:
  source_line: int | None
  output_line: int | None
  span: str | None
  entity_id: str | None
  json_path: str | None
```

`FeedbackRecord` 是 annotator 和 repair strategy 的共同输入。它不能只存在于 prompt 文本里，必须能在 task detail、audit history 和 repair context 中被追踪。

### 5.6 AuditEvent

```python
AuditEvent:
  event_id: str
  task_id: str
  type: str
  actor: str
  timestamp: datetime
  payload: dict
```


## 6. 状态机设计

### 6.1 Task 状态

当前实现使用 7 个 task status（`core/states.py:TaskStatus`）：

| Status | 含义 |
|---|---|
| `pending` | 等待 worker claim |
| `annotating` | 标注 LLM 调用进行中 |
| `qc` | 验证已通过，QC 进行中（或恢复中） |
| `arbitrating` | 仲裁 LLM 调用进行中，或 mechanical retry 等待重新 pickup |
| `accepted` | 终态 — 标注通过所有检查 |
| `human_review` | 准终态 — arbiter 真正不确定，或 retry 上限触发 |
| `rejected` | 保留给手动 reject |

整体流向：

```
PENDING ─┬─ (prelabel shortcut, current_attempt=0) ───────► QC
         └─► ANNOTATING ─► (validation) ─► QC ─► ACCEPTED
                          │                  │
                          └─► PENDING ◄──────┘  (retry, round_count++)
                                       │
                         (round_count ≥ max_qc_rounds)
                                       │
                                       ▼
                                  ARBITRATING ─┬─► ACCEPTED
                                               ├─► ARBITRATING (mechanical retry)
                                               └─► HUMAN_REVIEW
```

### 6.2 状态转换原则

- 所有状态转换走 `core/transitions.py:transition_task`，返回 `AuditEvent`
- worker 不直接写 task 文件；通过 `SqliteStore.save_task` + `append_event` 一并落地
- 每次转换必须有 reason + stage + metadata，落 `audit_events` 表
- 运行时状态（lease, active_run）单独建模，不污染业务 status

### 6.3 运行时状态

运行时状态以两张表表达：

```python
RuntimeLease:        # core/models.py
  lease_id: str
  task_id: str
  worker_id: str
  acquired_at: datetime
  expires_at: datetime

ActiveRun:
  run_id: str
  task_id: str
  worker_id: str
  stage: str
  provider_target: str
  started_at: datetime
```

设计理由：

- task 表只记业务真相（status + current_attempt + metadata），不混入活跃进程注册
- lease 过期 / active_run 孤儿可独立检测，crash recovery 路径清晰
- scheduler 可重启而不丢业务状态：smart resume 用 task status + 工件存在性恢复执行位置


## 7. 存储架构

### 7.1 默认实现：SqliteStore

`SqliteStore` (`annotation_pipeline_skill/store/sqlite_store.py`) is the
authoritative metadata store. Every workspace contains:

- `db.sqlite` — task / event / attempt / feedback / outbox / lease / document
  metadata in 13 tables (see `store/schema.sql`), plus two additive tables
  added via `_ADDITIVE_MIGRATIONS_SQL`: `entity_conventions` (high-trust
  prompt-injection dictionary) and `entity_statistics` (per-(project, span)
  type-frequency counter used by the prior verifier — see §11.9). WAL mode,
  single-machine multi-process safe; per-thread connections via
  `threading.local()` so the threaded HTTP dashboard can share the store
  safely.
- `artifacts/` — annotation result files referenced from `artifact_refs.path`.
- `document_versions/<doc>/<version>.md` — guideline content; DB row stores
  path + sha256.
- `exports/<export_id>/` — export output trees referenced from
  `export_manifests.output_paths_json`.
- `runtime/` — heartbeat, cycle stats, latest snapshot (file-only; low volume).
- `backups/` — periodic SQLite snapshots and the genesis JSON archive.

Recovery: WAL handles in-process crash safety; `apl db backup` produces
point-in-time snapshots; the pre-migration JSON tree is permanently archived
under `backups/genesis-YYYYMMDD/` and is the from-zero ground truth.

### 7.2 存储原则

- task 是 canonical business state
- event 是 append-only audit trail
- artifacts 与 task 解耦，只通过引用挂接
- media previews 是 artifact，不是业务状态；重新渲染 preview 不应改变 task 状态
- feedback 是 append-only 修复输入，不应被覆盖；新的 QC/validation 结果只能追加或 supersede 旧反馈
- runtime records 可被重建，不应成为业务真相唯一来源
- settings 是调度和 provider routing 的 canonical config，不写入 provider secret 明文
- external outbox 是外部状态回传的可靠队列，不能只依赖同步 HTTP 调用成功

### 7.3 原子性要求

- task 写入必须原子化
- event 追加必须尽量单向、不回写
- artifact 输出完成前先写临时文件再 rename
- 对单 task 更新加细粒度锁


## 8. 插件接口设计

### 8.1 DatasetAdapter

负责：

- 读取源数据
- 产出切片
- 生成 source manifest
- 构造 merge 所需 source key

接口建议：

```python
class DatasetAdapter(Protocol):
    def discover_sources(self, config: dict) -> list[SourceRef]: ...
    def build_tasks(self, source: SourceRef, task_size: int) -> Iterable[TaskDraft]: ...
    def build_manifest(self, draft: TaskDraft) -> dict: ...
```

### 8.2 PromptBuilder

负责：

- 生成 annotation prompt
- 生成 QC prompt
- 生成 repair prompt
- 将 open feedback records 压缩成 annotator 可执行的 compact feedback bundle

接口建议：

```python
class PromptBuilder(Protocol):
    def build_annotation_prompt(self, context: AnnotationContext) -> str: ...
    def build_qc_prompt(self, context: QcContext) -> str: ...
    def build_repair_prompt(self, context: RepairContext) -> str: ...
    def build_feedback_bundle(self, records: list[FeedbackRecord]) -> str: ...
```

### 8.2.1 AnnotatorSelector

负责：

- 根据 task manifest 的 modality、annotation requirements、artifact kind 选择 annotator
- 校验 annotator profile 是否支持目标输入和输出 artifact
- 在 primary annotator 不可用时选择 fallback annotator 或人工队列
- 记录选择原因和 capability match 结果

接口建议：

```python
class AnnotatorSelector(Protocol):
    def select(self, task: Task, profiles: list[AnnotatorProfile]) -> AnnotatorSelection: ...
```

选择逻辑必须基于结构化 manifest 和 profile 能力声明，不能基于 task 文本硬编码关键词。

### 8.3 Validator

负责：

- schema 校验
- deterministic lint
- merge gate

接口建议：

```python
class Validator(Protocol):
    def validate_output(self, task: Task, artifact: ArtifactRef) -> ValidationResult: ...
```

### 8.4 QcPolicy

负责：

- 抽样策略
- 通过阈值
- verdict 计算
- 基于结构化 QC 结果输出 `human_review_required` 和 review reason
- 支持 pipeline 强制 review 与 QC risk review 的合并决策

### 8.5 RepairStrategy

负责：

- 基于 validation/QC 结果决定 patch、rerun 或 escalate
- 基于 feedback records 选择 `bulk_code_repair`、`annotator_rerun`、`manual_annotation` 或 `reject`
- 为 annotator rerun 生成 repair context
- 为 bulk repair 返回 deterministic patch plan 或 repair artifact

### 8.6 MergeSink

负责：

- 将 accepted 结果写入目标系统
- 返回 merge report

### 8.6.1 PreviewRenderer

负责多模态 annotation artifact 的可视化预览：

```python
class PreviewRenderer(Protocol):
    def render(self, task: Task, source: ArtifactRef, annotation: ArtifactRef) -> ArtifactRef: ...
```

MVP 可以先实现 `ImageBoundingBoxRenderer`：

- 输入：`image_source` + `image_bbox_annotation`
- 输出：`image_bbox_preview`
- 坐标系统、图像尺寸、label 和 confidence 必须写入 metadata
- renderer 只生成 preview artifact，不决定 task 是否进入 QC

### 8.7 ProviderClient

负责：

- 调用具体模型供应商
- 返回结构化执行结果

Provider 不应感知 task store 和业务状态机。

### 8.8 ExternalTaskAdapter

负责从外部任务系统获取任务、提交结果和回传状态：

```python
class ExternalTaskAdapter(Protocol):
    def pull_tasks(self, limit: int) -> Iterable[ExternalTaskEnvelope]: ...
    def acknowledge_task(self, external_ref: ExternalTaskRef) -> None: ...
    def post_status(self, external_ref: ExternalTaskRef, status: ExternalTaskStatus) -> None: ...
    def submit_result(self, external_ref: ExternalTaskRef, result: ExternalTaskResult) -> None: ...
```

设计要求：

- 所有请求必须使用 idempotency key。
- 外部 API 错误不能丢失内部状态转换；失败状态回传写入 outbox 并由 retry drain 处理。
- adapter 返回的是外部 envelope，内部 task 仍由 `ExternalTaskService` 调用 `TaskFactoryService` 创建。
- MVP 只支持 pull + status callback + submit result；webhook ingestion 留到后续版本。

### 8.9 ProviderRegistry 和 StageRouter

Provider 配置分两层：

```python
ProviderConfig:
  provider_id: str
  kind: str
  models: list[str]
  default_model: str
  effort_options: list[str]
  secret_ref: str | None
  enabled: bool
  metadata: dict

StageRoute:
  stage: str
  primary_provider_id: str
  primary_model: str
  primary_effort: str | None
  fallback_provider_id: str | None
  fallback_model: str | None
  fallback_effort: str | None
  fallback_delay_seconds: int
  pause_until: datetime | None
  pause_reason: str | None
```

`StageRouter` 根据 stage、settings、task binding、provider pause 状态选择 route。已经绑定会话的 task 可以要求继续使用同一 provider；这种约束必须作为 route decision reason 写入 audit event。


## 9. 应用服务设计

### 9.1 TaskFactoryService

职责：

- 调用 adapter 创建 task draft
- 写入 raw slice / manifest
- 初始化 task

### 9.2 PipelineService

职责：

- 推进 task through stages
- 与 runtime backend 协作分配执行
- 汇总插件结果并决定下一状态
- 调用 `AnnotatorSelector` 选择合适 annotator
- 对启用 Human Review 的 task，在 QC 后进入人工复核阶段

### 9.3 RetryService

职责：

- 区分运行时错误和业务错误
- 根据 policy 生成 retry schedule
- 生成 repair context

### 9.3.1 FeedbackService

职责：

- 从 validator result、QC artifact、merge gate 和 human review 中生成 `FeedbackRecord`
- 读取当前 open feedback records，并按 task、attempt、severity、code 汇总
- 将旧反馈标记为 applied、dismissed 或 superseded
- 为 annotator rerun 构建 compact feedback bundle
- 为看板提供 feedback history read model
- 记录 operator 对 repair decision 的 override audit event

### 9.4 DashboardService

职责：

- 汇总 task 状态
- 提供 operator 所需最小视图
- 构建 dashboard read model
- 刷新 runtime overlay，避免缓存 snapshot 掩盖真实 worker 状态
- 输出 task detail payload，包括 attempts、events、artifacts、feedback records、provider route、external ref
- 输出 annotator profile、capability match、media preview artifact 和 Human Review 状态

### 9.5 SettingsService

职责：

- 读取 scheduler 设置：并发、每周期启动上限、自动派发开关
- 读取 provider registry、stage routes 和 annotator profiles
- 校验 provider/model/effort 是否属于可用选项
- 校验 annotator capability 是否满足 task requirements
- 提供 provider connectivity test 和 route validation
- MVP 不通过 UI/API 写入 provider、stage route 或 annotator YAML

### 9.6 ExternalTaskService

职责：

- 调用 `ExternalTaskAdapter.pull_tasks`
- 将外部 envelope 映射为 `TaskDraft`
- 创建带 `ExternalTaskRef` 的内部 task
- 在 stage transition 后写入 external status outbox
- 在 accepted / rejected / merged 后提交结果或失败原因
- 处理外部 API 幂等冲突、临时失败和 dead-letter


## 10. Runtime 设计

### 10.1 LocalRuntimeScheduler

当前实现是 `runtime/local_scheduler.py:LocalRuntimeScheduler` —— 单进程多 async
worker 的本地调度器：

- N 个 async worker（`max_concurrent_tasks`，默认 24）共享一个 SubagentRuntime
- 每个 worker 循环 claim → 跑 → 释放 lease
- 没有外部队列、没有 systemd、没有额外基础设施
- 适合单机跑 ~10K-100K task 的项目

### 10.2 Worker 循环

```python
loop:
  task, lease, run = try_claim_task(stage_target)
  try:
    await wait_for(runtime.run_task_async(...),
                   timeout=worker_task_timeout_seconds)
  except TimeoutError | Exception:
    pass  # 错误已记录在 attempt 行；worker 继续
  finally:
    delete_lease(); delete_active_run()
    if task.status == ANNOTATING:
      reset to PENDING  # "worker bailed mid-annotation"
```

`worker_task_timeout_seconds` 是单次 task 的硬上限。LLM 调用挂死（codex
subprocess 卡住、HTTP stream 不返回）超过这个时间会被取消，task 回收。

#### Stage priority

`_try_claim_task` 在扫候选时按 stage 优先级 + `created_at` 排序：

| 优先级 | Status | 理由 |
|---|---|---|
| 0 | ARBITRATING | 最靠近 terminal，优先消化避免堆积 |
| 1 | QC (resume) | 标注已完成，下一步就出结果 |
| 2 | ANNOTATING | restart orphan，可能升 QC 或回 PENDING |
| 3 | PENDING | 池子通常最大（千级），优先级最低 |

不加优先级时 PENDING 池子（常态 3000+）会一直霸占新 claim，几十个
ARBITRATING / QC 永远轮不到 → 累积成 ghost。`ARBITER_SLOT_FRACTION = 0.5`
保证一半 worker 留给 PENDING/ANNOTATION，防止 ARBITRATING 反向饿死下游。

### 10.3 Worker-bail reset

worker `finally` 段会主动把 ANNOTATING 的 task reset 回 PENDING。理由：

LLM 调用 raise 后（rate limit、网络、parse fail），task 留在 ANNOTATING 但没
lease。下一轮 claim 会触发 smart resume → 检查工件 → 没工件 → reset PENDING →
重新 claim → 又走 PENDING→ANNOTATING → LLM 又 fail → 死循环（实测 ~700
audit events/min）。在 finally 显式 reset 把这个环切断：下次 claim 看到的就是
干净的 PENDING。

### 10.4 Smart resume

`_try_claim_task` 在每次 claim 时检查 task 状态决定执行入口：

| Status seen | 有 annotation_result? | 动作 |
|---|---|---|
| ANNOTATING | yes | 升级到 QC + `runtime_next_stage=qc`（跳过重新标注） |
| ANNOTATING | no | reset 到 PENDING |
| QC + `runtime_next_stage=qc` | — | claim，从 QC 恢复 |
| ARBITRATING（无 lease） | — | claim，跑 `_run_rearbitration` |
| PENDING | — | claim，全 pipeline |

这样 runtime 重启后能从中断处继续，而不是从头跑。

### 10.5 启动时不动 in-flight task 状态

scheduler 启动时只清 stale lease（`_clear_stale_records`），**不**改 task
status。ANNOTATING / QC / ARBITRATING 都保留原状，下次 claim 通过 smart
resume / rearbitration 路径恢复执行。

之前版本有 `_recover_arbitrating_zombies` 把启动时所有 ARBITRATING 路由到
HR —— 这跟当前 arbiter 规则冲突（ARBITRATING 是合法的 mechanical retry 状
态），已移除。per-task 的 `arbiter_mechanical_retries` counter 仍然控制重
试上限，重启不影响。

### 10.6 Local CLI 调用约束

`provider: local_cli` 类型的 profile（codex / claude）通过 subprocess 调用，
runtime 强制以下隔离参数：

- `--ignore-user-config` —— 不受 user config 干扰
- `--ignore-rules` —— 跳过 user 安装的 rule files（skills、AGENT.md）
- `--ephemeral` —— 无 thread 持久化
- `--disable apps --disable plugins` —— 不加载外部集成
- `--config enabled_tools=[]` —— 抑制 tool use（arbiter 是纯 JSON 输出，不
  需要 bash/read）
- `--dangerously-bypass-approvals-and-sandbox --skip-git-repo-check` —— 非
  交互运行
- `claude --bare` —— 强制不读 OAuth / keychain / `~/.claude` credentials，
  不加载 hooks / CLAUDE.md / MCP / background prefetch

每次调用创建一个隔离的 `CODEX_HOME` / `HOME`（auth + config 拷贝），避免并
发 codex/claude 调用互相污染。

#### Subprocess lifecycle

`_die_with_parent()`（preexec_fn）调用 Linux `PR_SET_PDEATHSIG = SIGKILL`，
确保 runtime 进程被 kill 时所有 codex/claude 子进程立即被内核收割。没有这
个机制时，runtime 被 SIGKILL 后子进程会变成 PPID=1 的孤儿继续跑，可能写回
真实的 OAuth credentials 文件造成污染（曾观察到 isolated home 里
`auth.json` 被孤儿进程覆盖成测试用的 `{"token":"demo"}` 而触发后续大批 HR
的事故）。

#### OAuth state 拷贝 / 注入

`isolated_codex_home`：默认 path（profile 没设 `api_key`）下从用户 `~/.codex/`
拷贝 OAuth 完整 state 到 isolated HOME，包括：

| 文件 | 用途 |
|---|---|
| `auth.json` | `{auth_mode, OPENAI_API_KEY?, tokens{id,access,refresh}, last_refresh}` |
| `config.toml` | 先拷贝以保留 mcp 等配置，随后由 `_write_isolated_codex_config` 覆盖 model 字段 |
| `credentials.json` | legacy key-mode（少见） |
| `.credentials.json` | 某些版本用的隐藏 credentials 文件 |
| `installation_id` | OAuth client 标识；refresh flow 部分场景需要 |

**不拷贝**：`history.jsonl`、`log/`、`logs_2.sqlite*`、`cache/`、`.tmp/`（用户
历史与运行时状态）；`app-server-control/`、`app-server-daemon/`（IPC socket
目录）；`memories/`（用户记忆）。

`isolated_claude_home`：claude `--bare` 模式下，二进制不读 `.credentials.json`
里的 OAuth token（实测 strace 确认会 open 文件但忽略），所以拷贝文件无效。
改为**从 `~/.claude/.credentials.json` 读取 `claudeAiOauth.accessToken`，注入
到 isolated env 的 `ANTHROPIC_API_KEY`** —— 让 Claude Code Pro / Max 订阅
用户可以直接用 `claude_sonnet` / `claude_haiku` profile（无需 sk-ant-... API
key）跑 arbiter / QC，走的是订阅 quota 不是 metered API。优先级：profile.
api_key > OAuth fallback > 不注入（runtime 报正常 auth 错）。

第三方 Anthropic-兼容 vendor（DeepSeek / GLM / MiniMax）的 profile 必须显
式设 `api_key` 和 `base_url`，走第三方端点；不会触发 OAuth fallback。

### 10.7 本地 qwen / gemma 模型的路径翻译链

本地模型（`qwen3.6-35b-a3b`、`qwen3.6-27b`、`gemma-4` 等）通过 vLLM 服务，
但 runtime 用的是同一个 `claude_cli` provider 来调用——即 worker 子进程是
`claude --bare`，base_url 指向本地 gateway。请求要在两层之间翻译协议，每一
跳的 path 都不同：

```
worker (claude --bare, profile.base_url=http://127.0.0.1:8900)
    │
    │  POST /v1/messages?beta=true           ← Anthropic Messages API
    │  (claude CLI 总是发这个；beta=true 开 prompt caching 等扩展)
    ▼
gateway :8900  (Projects/llm-gateway/gateway.py)
    │  透明反代——path / headers / body 不动，唯一目的是给 LiteLLM
    │  套一层稳定的 base_url 并集中处理 /v1/ocr 等特殊端点
    │
    │  POST /v1/messages?beta=true
    ▼
litellm :8901  (Projects/llm-gateway/config.yaml)
    │  Anthropic → OpenAI 协议翻译。litellm_params.model 是 openai/qwen3.6-35b-a3b
    │  → openai provider 默认调 chat completions（不是 responses）
    │
    │  POST /v1/chat/completions
    ▼
model_manager :8002  (Projects/llm-gateway/model_manager.py)
    │  sticky routing 层。从 request header / body 抽 task_id（`x-task-id`
    │  或 OpenAI 标准 `user` 字段），把同一 task 的所有 turn 钉到同一个
    │  vLLM slot，让 prefix cache 在多轮 annotation/QC 对话间复用
    │
    │  POST /v1/chat/completions    (透传给选中的 vLLM slot)
    ▼
vLLM :9000  (RedHatAI/Qwen3.6-35B-A3B-NVFP4 等)
```

#### 关键认知

- **`/v1/responses` 不在这条链路上**。vLLM 本身支持，model_manager 也有专门
  的 `_handle_responses_api` 处理，但 LiteLLM 的 `openai/` provider 默认翻译
  到 `/v1/chat/completions`。要切到 Responses API 需要换 provider 类型（如
  `openai_responses/...` 或自定义 hook）。
- **sticky routing 走 task_id header，不走 `previous_response_id`**。后者要
  Responses API 才有；前者是 model_manager 自定义的协议，由
  `LLMGenerateRequest.task_id` 注入到 `ANTHROPIC_CUSTOM_HEADERS` 的
  `x-task-id` header（commit `68c272d`）。同一 task 的 annotation → QC →
  arbitration 多轮调用会命中同一 vLLM slot，KV cache 命中率从 0% 升到稳态
  >50%。
- **协议层的 `beta=true`** 是 claude CLI 自己加的。LiteLLM 接得住，会把对应
  的 `anthropic-beta` header 转成 OpenAI 端没有概念的功能（如 prompt
  caching）做 best-effort 适配。

#### 何时这条链路出问题

- gateway 起来但 litellm 没起：每个 POST 返 500，gateway log 里只看到
  HEAD / 健康探活。
- litellm 起来但 model_manager 没起：litellm 5xx；access log 里能看到
  upstream connection refused。
- model_manager 起但 vLLM slot 全冷：model_manager 选择 spawn 而非 503，
  worker 看起来"卡住"几十秒（vLLM 启动 + 加载权重耗时）。
- 本地 API key 错（`local-qwen36` mismatch）：vLLM 直接返 401，沿链路传回
  worker，新 instrumentation 会捕获并写 `provider_alert` 到 alerts.jsonl。


## 11. 执行模型

当前实现是三 LLM 角色协作的循环（`runtime/subagent_cycle.py:SubagentRuntime`）：

| Role | 默认 profile | 职责 |
|---|---|---|
| annotator | `minimax_2.7` | 从原始输入产出结构化标注 |
| QC | `deepseek_flash` | 找标注的缺陷，写 feedback |
| arbiter | `codex_5.5_arbiter` (codex CLI, gpt-5.5) | 仲裁分歧、产出修正 |

外加一个 `fallback` target（`codex_5.4_mini`）供主路 429 时透明切换。

### 11.1 Annotate 阶段

1. worker claim 一个 PENDING task
2. 检查 prelabel shortcut：`metadata.prelabeled=true` AND `current_attempt=0`
   AND 已存在 `annotation_result` artifact → 直接复用，跳到 QC
3. 否则调用 annotator LLM，写 `annotation_result` artifact
4. transition → ANNOTATING

### 11.2 Validate 阶段

annotation 写完后立刻跑两层确定性检查：

1. **Schema 校验** —— `core/schema_validation.py:validate_payload_against_task_schema`
   读项目 `output_schema.json` 或 task `annotation_guidance.output_schema`
2. **Verbatim 校验** —— `find_verbatim_violations`：所有 entity / json_structures
   span 必须是 `input.text` 的精确子串

任一失败 → 写 BLOCKING `FeedbackRecord(source_stage=VALIDATION)` → task 回
PENDING 重试。无新工件，annotation 留作历史记录。

Verbatim guard 在三个写入路径都有：annotator 输出、arbiter 修正、operator
人工修正。否则 5% 抽样发现 ~11% accepted task 含幻觉 span。

### 11.3 QC 阶段

1. validation 通过后 transition → QC
2. QC LLM 看到当前 annotation + 所有 open feedback + annotator discussion
   replies（如有）
3. QC 输出：
   ```json
   {"passed": true | false,
    "message": "...",
    "failures": [
      {"category": "missing_phrase|...",
       "message": "...",
       "confidence": "certain|confident|tentative|unsure",
       "target": {...}}
    ],
    "consensus_acknowledgements": ["feedback_id", ...]}
   ```
4. `passed=true` → ACCEPTED
5. `passed=false`:
   - `consensus_acknowledgements` 关闭对应 feedback（QC 看了 annotator
     的反驳后承认）
   - `failures` 开新 FeedbackRecord
   - task 回 PENDING，下一轮 annotator 拿到新 feedback bundle 重写

### 11.4 重试循环

每轮 = 一次 annotator + 一次 QC。`round_count` = task 上 *open* QC/validation
feedback 数量（consensus 关闭的不算）。

```
PENDING → annotator → validation (fail → PENDING)
                    → QC (passed → ACCEPTED, failed → PENDING + feedback)

if round_count >= max_qc_rounds (默认 3) → ARBITRATING
```

### 11.5 Arbiter

输入：input task + 最新 annotation + 所有 open feedback + annotator
discussion replies。

输出：
```json
{
  "verdicts": [
    {"feedback_id": "...",
     "verdict": "annotator|qc|neither",
     "confidence": "certain|confident|tentative|unsure",
     "reasoning": "..."}
  ],
  "corrected_annotation": {"rows": [...]} | null
}
```

`verdict`:
- `annotator` —— QC 投诉错了，当前 annotation 留下
- `qc` —— annotation 错了，按 QC 说的修
- `neither` —— 都不对，arbiter 给出正确版本

`confidence` 是 4 档语言标签（弃用数字 —— 经过校准实测，所有数字 bucket 的
正确率几乎一样，是噪声）。

#### 内部 retry loop

`_arbitrate_and_apply` 自带最多 `arbiter_verbatim_retries`（默认 2）次重试：

- arbiter 给了 qc/neither 高 confidence 但 `corrected_annotation=null` ——
  显式提示"你忘了 JSON" 让它重出
- arbiter 的 corrected 含非 verbatim span —— 给出具体哪个 span 错了，要求重
  emit verbatim

retry 耗尽后清空 corrected_annotation，让外层逻辑继续。

#### Safe-span auto-fix

verbatim 校验之前，`auto_fix_safe_spans_in_place`（`core/schema_validation.py`）
对 corrected_annotation 做安全字符级修正，三类操作都保证**不改字母 / 数字 /
任何非空白字符**：

- **头尾 trim**：去掉 span 两端的空白 / 句末标点 / 引号（中英文都支持）
- **Internal whitespace recovery**：当 span 是 input 的"去空白版"时
  （例：arbiter 输出 `"1764"`，input 是 `"1 7 6 4"`），自动恢复 input 里
  那段原始带空白的子串。处理 CJK OCR / 排版风格的字符间空格场景
- **共享类型 field routing**：单 word span 放 `entities.<type>`，多 word
  放 `json_structures.<type>`

只有真正的字母级差异（大小写、简繁体、Unicode 等价、paraphrase）才会触发
verbatim 失败并走 retry / bail counter。

#### Arbiter outcome counters

四个计数器驱动后续决策：

| Counter | 在何时 +1 |
|---|---|
| `closed` | verdict=`annotator`, label ∈ {certain, confident} |
| `fixed` | verdict ∈ {`qc`, `neither`}, label confident/certain, AND retry 后 `corrected_annotation` 仍非空 |
| `unresolved` | label ∈ {tentative, unsure, None} |
| `mechanical_fail` | verdict ∈ {`qc`, `neither`} 高 confidence 但 `corrected_annotation` 是 null（retry 耗尽）；OR 未知 verdict 值 |

### 11.6 HR 路由规则

只有"arbiter 真正不确定"才进 HR。所有机械故障（codex error、缺 fix、
verbatim 违规、JSON parse fail）都是 mechanical retry，留在 ARBITRATING 等下
次 worker pickup。

```
_terminal_from_arbiter:
  unresolved > 0       → None  (上层决定：HR 还是 retry)
  fixed > 0 + valid    → ACCEPTED  (写修正、accept)
  closed > 0           → ACCEPTED  (annotator 的 annotation 留下)
  else                 → None  (mechanical 信号)

caller (validation / qc / rearbitration paths):
  terminal is not None              → ACCEPTED (已经 transition)
  terminal is None, unresolved > 0  → HUMAN_REVIEW
  terminal is None, unresolved == 0 → 留 ARBITRATING (mechanical retry)
                                       └─► after N=3 retries → HUMAN_REVIEW
```

#### Mechanical retry cap

`SubagentRuntime.ARBITER_MECHANICAL_RETRY_CAP = 3`。每次 mechanical fail
自增 `task.metadata.arbiter_mechanical_retries`。到 3 强制 → HR，reason 带
"arbiter exhausted N mechanical retries without an actionable verdict"。
counter 持久化在 task metadata，重启 runtime 不丢。

为什么 mechanical retry 留 ARBITRATING 而不是回 PENDING：annotation 没变，
重跑 annotator 是浪费 —— 只需要 arbiter 再判一次。scheduler 的 claim
逻辑会自动 pick up 无 lease 的 ARBITRATING task。

### 11.7 Human Review

进入 HR 的路径：
- arbiter `unresolved > 0`（真不确定）
- mechanical retry cap 触发（3 次都失败）
- scheduler init 的 zombie recovery（ARBITRATING + 无 lease）
- operator 在 HR drawer 主动 reject（`HumanReviewService`）

HR 卡片在看板上：
- 显示 reason quote（最近一次 transition 的 reason）
- 自动失败的卡显示 detail 段：`!arbiter_ran` / `unresolved>0` / generic
- operator 可以 accept、reject、submit corrected answer、或拖回 Arbitration

operator submit_correction 也走 verbatim guard + schema 校验，不通过会直接
拒绝写入。

### 11.8 Rearbitrate

Operator 把 HR/REJECTED 卡拖到 Arbitration → API 把 status 改成
ARBITRATING。Scheduler claim 后跑 `_run_rearbitration`：

- 同样调 `_arbitrate_and_apply`，但 `require_rebuttal=False` +
  `include_closed_feedbacks=True`
- 即使 annotator 没写 discussion reply 也能跑 arbiter
- 决策逻辑同 §11.6（HR 只在 unresolved > 0 时）

### 11.9 Prior-driven verifier（V1.2）

理论背景：多 agent LLM 标注存在系统性 correlated error（实测
GPT-4/Claude/Gemini 之间 forecasting error 相关性 r=0.78）。Annotator+QC
consensus 不是独立投票，arbiter 是另一个 LLM，纯 LLM 聚合规则在没有外部
verifier 时无法可靠避开 cascade。Verifier 用项目内历史决策的经验分布作为
**外部验证信号**，独立于 LLM 决策本身。

详细设计：`docs/superpowers/specs/2026-05-17-prior-driven-verifier-design.md`。

**双表分工**:

| 表 | 输入 | 用途 |
|---|---|---|
| `entity_statistics` | ALL ACCEPTED（含 arbiter），HR 加权 5x | verifier 查 prior |
| `entity_conventions`（已存在） | QC consensus + verifier agree，或 HR 决定（**不含** arbiter） | 注入 prompt |

`entity_conventions` 严格排除 arbiter 是因为 arbiter 是 LLM，如果错误进
prompt 会 self-reinforcement。`entity_statistics` 接受 arbiter 因为它是
empirical distribution 不会被 prompt 看到，cascade 路径被切断。

**Verifier 语义** (`PriorVerifier.check`)：
- `total < 10` → `cold_start`（不动）
- 主类型 < 80% → `agree`（prior 不够独断）
- 主类型 == proposed_type → `agree`
- 否则 → `divergent`

**三个触发点**:

1. **QC pass**: annotator+QC consensus 后查 verifier。`divergent` →
   路由到 ARBITRATING + 写 `prior_disagreement` BLOCKING feedback。
   `agree`/`cold_start` → ACCEPTED + stats++（`agree` 时 conventions++）。

2. **Arbiter ruling**: arbiter 出 verdict 后 post-check 一次。`divergent`
   → 标记 task metadata，scheduler 下个 cycle pick up 后调 **第二个
   arbiter**（不同 model family，配置 `arbiter_secondary` target）。

3. **HR submit_correction**: 同样查 verifier，`divergent` 时 raise
   `SchemaValidationError`，UI 可让 operator 用 `force=True` 覆盖（human
   authority）。

**第二 arbiter 三向解析**:

| 第二 arbiter 选择 | 结果 |
|---|---|
| 与第一 arbiter 同 | ACCEPTED，两 LLM 推翻 prior |
| 与 prior 主类型同 | 改写 annotation，ACCEPTED 用 prior 类型 |
| 第三种 | HR（三方不一致需人裁定） |

**Posterior Audit tab**:
Operator 点 "Check" 后端扫所有 ACCEPTED：
- **Task-level deviations**: 当前 annotation 与 prior 分歧的 (task,
  span, type) 列表，每行 "Send to HR" 按钮
- **Contested spans**: prior 分布本身没共识的 span（≥10 样本，
  无类型 ≥80%，至少两类型各 ≥20%），每行 "Declare canonical type" 让
  operator 一次定调，写 `entity_conventions` + 加权 `entity_statistics`

实现入口：`annotation_pipeline_skill/services/entity_statistics_service.py`
和 `runtime/subagent_cycle.py` 的 verifier 注入点。


### 11.10 Annotation Knowledge Base（V1.3）

V1.2 把项目经验作为 runtime 自动调用的 verifier。V1.3 把同一份经验额外
暴露成 **agent 主动调用的 MCP tool**，让 annotator / QC 子 agent 在标注
前/标注中可以自己决定查询。完整设计：
`docs/superpowers/specs/2026-05-19-annotation-knowledge-base-design.md`。

#### 组件分层

```
annotator / qc 子 agent (claude --bare)
        │ MCP stdio (JSON-RPC over claude --mcp-config)
        ▼
annotation_pipeline_skill/mcp/kb_server.py
        │ 调用 (in-process)
        ▼
annotation_pipeline_skill/mcp/check_past_experience.py（纯函数）
        │
        ├─ EntityConventionService（已存在，读 entity_conventions）
        ├─ entity_conventions.proposals_json（扩展两字段）
        ├─ similarity.diverse.select_diverse_examples（新）
        ├─ similarity.minhash.shingle（扩展 CJK 路径）
        └─ text.wordfreq_utils.wordfreq_score（从 api.py 提取）
```

四层都遵守"零业务状态、查询时组合"的原则。`check_past_experience` 不写
任何东西；不维护 cache；不订阅事件——每次调用都重新读 SQLite 并 MinHash
当前 proposals。响应 dict 上限不超过 200 条 proposal 时 < 100 ms。

#### Schema 扩展（唯一的破坏性变更）

`entity_conventions.proposals_json` 里每条 proposal 新增 **两个可选字段**:

```python
{
  "type": str, "source": str, "task_id": str | None,
  "row_id": str | None,           # NEW
  "context_snippet": str | None,  # NEW: span 周围 ±80 字符的 row 片段
  "notes": str | None, "at": str,
}
```

因为 `proposals_json` 是自由 JSON blob，不需要 ALTER TABLE 也不需要回填。
旧 proposal 没有这两个字段——`check_past_experience` 把它们当作 `None`
处理，不影响 distribution 统计，只是没法贡献到 `examples_by_type`。

写入方面：`EntityConventionService.record_decision()` 新增两个 kwargs
（`row_id`, `row_content`），调用 `_build_context_snippet(span, row_content)`
生成 snippet。调用站点 (`subagent_cycle._record_conventions_from_qc_consensus`,
`human_review_service._record_conventions_from_correction`) 通过新增的
`extract_entity_type_decisions_with_row` 辅助函数把 row 信息带过去。
operator-declared 的两个写入点（api.py 的 Set Convention 端点和
posterior_audit_operator 路径）刻意不传——operator 在 UI 上声明 convention
时本来就没绑定到具体 row，传 `None` 是正确语义。

#### MCP Server 形态

`kb_server.py` 是 thin wrapper：argparse 解析 `--project-root` /
`--project-id`，构造一个 `Server("annotation-kb")`，注册唯一一个 tool
（input schema 只声明必填 `entry: string`），把 `tools/call` 转发给
`check_past_experience` 纯函数。返回值打成单个 `TextContent`（`type:
"text"` + JSON-encoded payload）。所有错误（`ValueError` 输入错误、
`sqlite3.OperationalError` 存储错误、`json.JSONDecodeError` 数据损坏）
都被翻译为 `{"error": "..."}` payload 而非 protocol exception，避免任何
错误 break MCP 通道。

进程模型：每个 annotator subprocess 启动时由 Claude CLI fork 一个独立
的 `python -m annotation_pipeline_skill.mcp.kb_server` 子进程，通过 stdio
通信，subprocess 退出时 Claude CLI 负责清理。SQLite 用单连接（设了
`check_same_thread=False`），asyncio 单事件循环驱动，不会出现跨线程访问。

#### Diversity Sampling 算法

`select_diverse_examples(snippets, k=3)` 用 Gonzalez farthest-first
traversal：

1. dedupe snippets（保持插入顺序）
2. 选 lex-smallest 作为 seed（确保 deterministic）
3. 循环 (k-1) 次：每次选 candidate i 使得
   `1 - max(jaccard(i, j) for j in selected)` 最大；tie-break 用 lex
   比较

MinHash 用 `num_perm=64`（低于 `row_dedup` 的 128，因为我们对 short
snippets 小规模 pairwise 比较）。Shingle 路径走 `similarity.minhash.shingle`
（接下来说明的 CJK gate）。

#### CJK Shingle Gate

`similarity.minhash.shingle()` 之前对所有输入做 word-level n-gram
（whitespace split）。CJK 文本因为没有空格被退化成单 shingle，使
Jaccard 二值化、`row_dedup` 实质失效、KB 多样性采样失效。

新增 `_CJK_RE = re.compile(r"[一-鿿㐀-䶿]")` gate：

```python
if _CJK_RE.search(normalized):
    import jieba   # lazy import — ASCII 项目不付加载成本
    tokens = [t for t in jieba.cut(normalized) if t.strip()]
else:
    tokens = normalized.split(" ")
```

ASCII 路径完全不变（实测对比："Apple's customer support..." 在统一
jieba 路径下会拆出 `apple / ' / s` 产生 6 个 3-gram，而 split 路径只
4 个，3 个共享——会破坏 row_dedup 的已校准 threshold）。CJK 路径从
"单 shingle 退化" 升级为 "jieba 词级 n-gram"，对 row_dedup 是一次有意
的精度提升（CJK-heavy 项目可能需要 re-verify `jaccard_threshold`，
spec Risks 里有说明）。

#### LLM Provider 切换

profile 三个新字段（`mcp_servers`, `strict_mcp_config`, `disallowed_tools`）
落到 `llm/profiles.py::LLMProfile` 上，validator (`_optional_mcp_servers`,
`_optional_string_list`) 拒绝结构错误的 yaml。

`llm/local_cli.py::_generate_claude` 在 `isolated_claude_home(...)`
context 内 materialize 一个 `mcp-config.json` 到 isolated home 目录，然后
重建 command（`build_claude_command` 接受三个新 kwargs：
`mcp_config_path`, `strict_mcp_config`, `disallowed_tools`）。临时文件
随 isolated home 一起被 GC，不需要额外 try/finally。

provider switch 走的是已经存在的 `LLMProfile.base_url` → `isolated_claude_home`
→ `subprocess.Popen(env={"ANTHROPIC_BASE_URL": ..., ...})` 路径。子进程
拿到独立 env 字典，parent process 的 `os.environ` 永远不被修改。

#### System-Level Prompt

`runtime/subagent_cycle.py::_annotation_instructions()` 和
`_build_qc_instructions()` 各嵌入一段 "KNOWLEDGE BASE TOOL:" 段落。措辞
是 conditional 的（`"when the mcp__annotation-kb__check_past_experience
tool appears in your tools list..."`），所以没装 MCP server 的 profile
看到同一段文字但 agent 自然 no-op。指导内容包括：何时调用（ambiguous
named entities）、如何用返回结果（active convention 优先；disputed 用
per-type examples 做 analogy；`generic_word` + low evidence 通常不标）、
何时跳过（明显的非实体、schema 一目了然的 span）。

#### Runtime 路径上的副效应修复

V1.3 重写 `_profile_name_for_target`：旧实现为了拿 `client.profile.name`
会构造一个 throwaway client 然后扔掉。在 production 几乎免费，但 finite-
list 测试 stub 里每次调用消耗一个 stub，破坏了 retry 流程的两个测试。
新实现把 profile-name 改成由 `_call_client` 副作用填的 cache
（key = `result.provider`，跟 `_write_pinned_handle` 写入的 minted-by
字段对齐）。无 probe，无 stub 消耗，pin-handle 跨 provider 校验语义
保持不变。

#### 测试矩阵

| 文件 | 测试数 | 范围 |
|---|---|---|
| `tests/test_text_wordfreq_utils.py` | 5 | `wordfreq_score` 行为 |
| `tests/test_similarity_minhash.py` | 11（+5 新增） | CJK gate + ASCII 不变 |
| `tests/test_similarity_diverse.py` | 5 | farthest-first 算法 |
| `tests/test_entity_convention_proposals_schema.py` | 10 | proposal 扩展 + `extract_*_with_row` |
| `tests/test_mcp_check_past_experience.py` | 7 | 纯函数所有分支 |
| `tests/test_mcp_kb_server.py` | 3 | stdio MCP 协议（含错误路径） |
| `tests/test_llm_profiles_mcp.py` | 5 | profile schema + validator |
| `tests/test_local_cli_claude_mcp.py` | 4 | `build_claude_command` MCP 标志 |

50/50 通过。E2E 验证（`docs/release/annotation-kb-verification.md`）走真实
Claude CLI + DeepSeek endpoint，证明 agent 在 disputed span 上自主调用
工具并根据 per-type examples 正确消歧。


## 12. 错误模型

所有 in-flight 错误都是非致命的，由三层兜住：

### 12.1 SubagentRuntime 层

每个 LLM 调用包在 try/except 里：raise → 尽量不记录 attempt（有些路径
`status=failed`），让上层把 task 留在 in-flight status，由 worker `finally`
清理。

### 12.2 Provider fallback

`SubagentRuntime._generate_async` 在每次调用前包一层：

```python
try:
    return await self._call_client(target, request)
except Exception as exc:
    if target == "fallback" or not _is_rate_limited(exc):
        raise
    return await self._call_client("fallback", request)
```

`_is_rate_limited` 识别 `openai.RateLimitError`、`status_code == 429`、
以及 "rate limit" / "429" / "too many requests" 字符串匹配（覆盖 local-CLI
client 抛出的非结构化异常）。

Try-first 语义 —— 每次都先打主路；只在 429 时 fallback。无 circuit breaker
/ recovery window，主路恢复就自动用回去。

### 12.3 Worker `finally` 兜底

worker 释放 lease + active_run，把 ANNOTATING reset 到 PENDING（参见 §10.3
worker-bail reset）。

### 12.4 Smart resume + zombie recovery

下次 claim 或下次 scheduler 启动，没到达终态的 task 走 §10.4 / §10.5 的恢复
路径。

### 12.5 错误进 HR 的唯一路径

- arbiter `unresolved > 0`（真不确定）
- arbiter mechanical retry cap（3 次都失败）
- operator 主动 reject

annotator / QC / validation 层的错误**不**直接进 HR，全部走 retry 循环 →
（达到 max_qc_rounds）→ ARBITRATING → arbiter 决定。
Scheduler 重启**不**把任何 task 路由到 HR —— in-flight task 由 smart resume
接管。


## 13. 配置架构

配置分两层：workspace-global 的 LLM profile，和每个项目的 workflow / annotator
/ schema。

### 13.1 配置层级

```text
<workspace>/llm_profiles.yaml          # 多项目共用，profile + targets

<project>/.annotation-pipeline/
  workflow.yaml                        # runtime + 调度策略
  annotators.yaml                      # 该项目的 annotator profile 定义
  external_tasks.yaml                  # 外部任务系统绑定
  callbacks.yaml                       # 回调 / outbox
  output_schema.json                   # JSON Schema —— 所有 accepted 必须符合
```

profile 在 scheduler 启动时加载到 registry 并冻结，YAML 改动需要重启
runtime 才生效。

### 13.2 llm_profiles.yaml 示例

```yaml
profiles:
  minimax_2.7:
    provider: openai_compatible
    provider_flavor: minimax
    model: MiniMax-M2.7
    base_url: https://api.minimaxi.com/v1
    timeout_seconds: 300
    api_key: <secret>

  deepseek_flash:
    provider: openai_compatible
    provider_flavor: deepseek
    model: deepseek-v4-flash
    base_url: https://api.deepseek.com
    timeout_seconds: 120
    api_key: <secret>

  codex_5.5_arbiter:
    provider: local_cli
    cli_kind: codex
    cli_binary: codex
    model: gpt-5.5
    reasoning_effort: high
    timeout_seconds: 900

  codex_5.4_mini:
    provider: local_cli
    cli_kind: codex
    cli_binary: codex
    model: gpt-5.4-mini
    reasoning_effort: medium
    timeout_seconds: 900

targets:
  annotation: minimax_2.7
  qc: deepseek_flash
  arbiter: codex_5.5_arbiter
  arbiter_secondary: claude_sonnet_arbiter   # V1.2 §11.9 — 不同 family，prior verifier 触发时调用
  fallback: codex_5.4_mini
  coordinator: glm_46

limits:
  max_concurrent_tasks: 16            # async worker 数（local_scheduler 直接读这个）
```

`targets` 是逻辑角色 → profile 的映射。runtime 通过
`registry.resolve("annotation")` 拿到当前 annotation 该用的 profile。
fallback 是 `_generate_async` 在 429 时切换的目标（参见 §12.2）。

`max_concurrent_tasks` 是 worker 协程数的硬上限（`local_scheduler.py:509`
`asyncio.create_task(worker()) for _ in range(self.config.max_concurrent_tasks)`）。
ARBITER_SLOT_FRACTION = 0.5 在此基础上限制并发 arbitration 数，保证
annotation/QC 不被 arbiter 长尾饿死（参见 §10.2）。

#### 真 Anthropic 模型的 OAuth fallback

对于 `model: claude-*` + `base_url: https://api.anthropic.com` 的 profile，
**可以省略 `api_key` 字段** —— runtime 会自动从 `~/.claude/.credentials.json`
里的 `claudeAiOauth.accessToken` 注入到 `ANTHROPIC_API_KEY`，走用户
Claude Code Pro / Max 订阅 quota（详见 §10.6 OAuth state 拷贝 / 注入）。

```yaml
claude_sonnet:
  provider: local_cli
  cli_kind: claude
  model: claude-sonnet-4-5
  base_url: https://api.anthropic.com
  # api_key 可省 —— 自动复用本地订阅 OAuth
```

第三方 Anthropic-兼容 vendor（DeepSeek / GLM / MiniMax 的 `/anthropic`
端点）必须显式设 `api_key`，不会触发 OAuth fallback。

### 13.3 workflow.yaml 示例

```yaml
runtime:
  max_concurrent_tasks: 24            # async worker 数
  max_qc_rounds: 3                    # 触发 arbiter 的 round 阈值
  worker_task_timeout_seconds: 900    # 单 task 硬上限
  arbiter_verbatim_retries: 2         # arbiter 内部 retry 次数
  prior_verifier:                     # V1.2, §11.9
    enabled: true
    min_prior_samples: 10             # cold_start 阈值
    dominance_threshold: 0.80         # 主类型占比阈值
    hr_weight: 5                      # HR 决策权重
  posterior_audit:
    min_contested_samples: 10
    min_runner_up_share: 0.20         # 至少两类型各 ≥20% 才算 contested

qc_policy:
  sampling: full                       # 还是 random / risk_based
```

### 13.4 annotators.yaml 示例

```yaml
annotators:
  default_text:
    display_name: Default text annotator
    modalities: [text]
    annotation_types: [extraction, classification]
    provider_target: annotation        # 引用 llm_profiles.yaml 的 target
    enabled: true
```

### 13.5 output_schema.json

JSON Schema (Draft 2020-12) 约束 annotator / arbiter / operator 的输出。
Verbatim 校验 (`find_verbatim_violations`) 额外约束 entity / json_structures
phrase 必须是 `input.text` 子串。这两层一起保证 accepted artifact 不含幻觉。


## 14. 接口设计

### 14.1 CLI

建议命令：

- `annotation-pipeline init`
- `annotation-pipeline create-tasks`
- `annotation-pipeline run`
- `annotation-pipeline retry --task-id ...`
- `annotation-pipeline inspect --task-id ...`
- `annotation-pipeline merge --task-id ...`
- `annotation-pipeline dashboard build`
- `annotation-pipeline dashboard serve`
- `annotation-pipeline settings validate`
- `annotation-pipeline settings set-route --stage annotation ...`
- `annotation-pipeline annotators add ...`
- `annotation-pipeline annotators select --task-id ... --annotator-id ...`
- `annotation-pipeline preview render --task-id ...`
- `annotation-pipeline human-review decide --task-id ... --decision accept`
- `annotation-pipeline feedback decide --task-id ... --feedback-id ... --decision annotator_rerun`
- `annotation-pipeline providers test --provider-id ...`
- `annotation-pipeline external pull --limit ...`
- `annotation-pipeline external drain-outbox`
- `annotation-pipeline doctor`

### 14.2 API

MVP 需要只读、少量控制、settings 和外部任务接入接口：

- `GET /health`
- `GET /dashboard`
- `GET /tasks/<task_id>`
- `GET /settings`
- `POST /settings/validate`
- `GET /providers`
- `POST /providers/test`
- `POST /tasks/<task_id>/retry`
- `POST /tasks/<task_id>/approve`
- `POST /tasks/<task_id>/reject`
- `POST /tasks/<task_id>/merge`
- `POST /tasks/<task_id>/start`
- `POST /tasks/<task_id>/stop`
- `GET /annotators`
- `POST /tasks/<task_id>/annotator`
- `GET /tasks/<task_id>/preview`
- `POST /tasks/<task_id>/preview/render`
- `POST /tasks/<task_id>/human-review/decision`
- `GET /tasks/<task_id>/feedback`
- `POST /tasks/<task_id>/feedback/<feedback_id>/decision`
- `POST /external/tasks/pull`
- `POST /external/tasks/status`

API 设计原则：

- 所有写接口都必须返回 audit event id。
- 写接口不能绕过 application service。
- MVP 中 `GET /settings`、`POST /settings/validate`、`GET /providers`、`GET /annotators` 是只读/校验接口；UI 不写 YAML 配置。
- 外部 task status endpoint 用于 webhook 或 adapter 测试，不作为 core 状态真相。
- MVP 不提供 webhook ingestion endpoint；外部任务进入系统必须通过 pull 或本地 import。
- dashboard API 返回 read model，其中 worker live counts 必须标明来源：runtime、snapshot 或 fallback。

### 14.3 Web Dashboard

默认 Web 看板必须使用 TypeScript Web 框架实现，推荐 React 系生态。主界面采用 Kanban-first 布局，后端仍由 Python API 提供 read model 和控制接口；前端只依赖 HTTP API，不读取 task store 文件，也不直接调用 provider。

- Sidebar：runtime health、heartbeat age、service state、并发设置、自动派发开关、provider route form、刷新按钮
- Summary：task 总数、ready/pending、active、live workers、done/merged
- Kanban board：按 pipeline stage 分列展示 task card，过滤搜索后仍保留列结构
- Default columns：Ready、Annotating、Validating、QC、Human Review、Repair、Accepted、Rejected、Merged
- Task card：status badge、task id、source/ref、row range 或 slice summary、runtime parts、retry time、QC history、错误摘要、start/stop/detail 控件
- Detail drawer：点击 task card 后打开右侧抽屉，承载 attempts、events、artifacts、feedback、provider route、annotator、preview 和 Human Review 控件
- Human Review：图片 bbox overlay、视频帧 overlay、点云 viewer 状态和 accept/reject/request-repair 控件
- Annotator：当前 annotator、可用 capability、fallback、recent quality metrics
- Filters：source、status、task id/external id 搜索
- Settings：阶段级 primary/fallback provider、model、effort、pause reason

前端实现要求：

- 使用 TypeScript 定义 dashboard、task detail、settings、provider route 和 action response 类型。
- API client 集中在 `web/src/api/`，不要在组件里散落 `fetch` 调用。
- 写操作必须使用后端返回的 audit event id 更新 UI 状态或触发重新拉取。
- provider secret 只显示引用状态，不能在前端 payload、local storage 或日志中出现明文。
- 控制动作必须有 disabled/loading/error 状态，避免重复提交。


## 15. 观测性

### 15.1 日志

分三类：

- operator log
- task event log
- worker execution log

### 15.2 Metrics

建议采集：

- tasks by status
- average stage duration
- validation failure rate
- qc failure rate
- feedback count by severity, code, source, and repair decision
- bulk repair success rate
- annotator rerun success rate after feedback
- retry count distribution
- merge success rate

### 15.3 Dashboard Snapshot

Dashboard 不直接重扫全部 artifacts，而是消费聚合 snapshot，降低开销并减少状态抖动。


## 16. 安全与隔离

### 16.1 Worker 隔离

即使默认 runtime 是 local subprocess，也应支持：

- 独立工作目录
- 独立临时目录
- 限制可写路径
- 最小权限 provider config 注入

### 16.2 密钥管理

- provider token 不写入 task state
- token 通过环境变量或 secret provider 注入
- artifacts 中避免落私密原始凭据

### 16.3 审计要求

- 所有人工 override 都应留下 audit event
- 所有自动 repair 都应保留输入依据


## 17. 测试策略

### 17.1 单元测试

覆盖：

- 状态机转换
- store 原子写入
- retry policy
- validator contract
- adapter contract
- settings validation
- provider route selection
- external task outbox behavior

### 17.2 集成测试

覆盖：

- create task -> annotate -> validate -> qc -> merge
- worker crash recovery
- retry scheduling
- merge failure recovery
- dashboard read model with runtime overlay refresh
- external task pull -> process -> status/result submission

### 17.3 合约测试

针对插件接口：

- DatasetAdapter contract tests
- Validator contract tests
- RuntimeBackend contract tests
- ProviderClient contract tests
- ExternalTaskAdapter contract tests

### 17.4 示例测试

仓库内 demo 项目必须在 CI 中可完整跑通。


## 18. 迁移与演进策略

### 18.1 从项目专用实现迁移到框架

迁移顺序建议：

1. 先定义 core models 和 plugin contracts
2. 再实现 file store + local runtime
3. 再把项目现有 validator、prompt builder、merge sink 包成 adapter
4. 最后按需接入 queued runtime 或 web UI

### 18.2 向后兼容原则

- 核心状态模型稳定优先
- 插件接口版本化
- runtime backend 可替换但不侵入 domain model


## 19. 推荐技术选型

### MVP

- Python 3.11+
- `pydantic` 或 dataclass + explicit validation
- 文件系统 store
- subprocess runtime
- Typer 或 argparse CLI
- FastAPI 或最小 HTTP server 提供 API
- TypeScript + React 系框架实现 Web dashboard

### 可选增强

- Redis 作为 queue backend
- SQLite/Postgres 作为 store backend
- Next.js、Vite React 或同等级 TypeScript 框架做更完整的 dashboard


## 20. 最终架构结论

`annotation-pipeline-skill` 的正确技术方向不是“把现有大脚本开源”，而是：

- 用清晰的核心领域模型重建任务系统
- 用插件接口隔离数据集和任务特定逻辑
- 用可替换 runtime 支持从单机到队列化部署
- 用 deterministic gate、QC、repair、merge 组成标准流水线

最终应形成一个“框架内核稳定、业务适配可插拔、默认本地可跑、重型部署可扩展”的开源 skill。
