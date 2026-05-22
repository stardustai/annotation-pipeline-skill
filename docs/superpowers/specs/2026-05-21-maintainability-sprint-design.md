# Maintainability Sprint — Design Spec

**日期**: 2026-05-21  
**范围**: 1–2 周，三条并行线  
**背景**: 全量 gap 分析（文档 vs 代码）发现三类系统性问题：operator 操作要依赖 Web UI、插件扩展没有正式接口、两份架构文档描述的内容和代码实际不符。本 spec 定义修复方案。

---

## 目标

1. **可用性（A）**: 补齐 operator 日常需要的 CLI 命令，降低对 Web UI 的依赖
2. **可扩展性（B）**: 声明 Plugin Protocol 接口，提取两个真实实现，让扩展路径清晰
3. **可维护性（C）**: 把 `subagent_cycle.py` 中职责最清晰的两块提取为独立模块；更新两份文档对齐代码现实

---

## 三条线概览

| 线 | 主要产出 | 预计周期 |
|---|---|---|
| 线 1：CLI 补齐 | `serve` 修复 + 4 个新命令 | 2–3 天 |
| 线 2：Plugin Protocol | `plugins/` 声明 + adapter 参考实现 + 两处提取 | 3–4 天 |
| 线 3：文档对齐 | TECHNICAL_ARCHITECTURE 三章节 + PRODUCT_DESIGN §10.1 | 2 天 |

三条线独立、互不阻塞，各自单独 PR 合入。


---

## 线 1：CLI 补齐

### 1.1 `serve` 注册修复（bug 级）

**问题**: `handle_serve` 函数存在，但 `build_parser()` 里从未 `add_parser("serve")`，导致命令无法从 CLI 入口调用。

**修复**: 在 `build_parser()` 的 subparsers 区段添加：

```python
serve_parser = subparsers.add_parser("serve")
serve_parser.add_argument("--project-root", type=Path, default=Path.cwd())
serve_parser.add_argument("--port", type=int, default=8765)
serve_parser.set_defaults(handler=handle_serve)
```

### 1.2 `inspect <task_id>`

**用途**: 在终端查看某个 task 的当前状态，不需要打开看板。

**命令**:
```
apl inspect <task_id> [--project-root PATH]
```

**输出内容**:
- task status、created_at、updated_at
- current_attempt 编号
- 最近 3 次 attempt 摘要（stage / provider / status / error summary）
- 所有 open FeedbackRecord（code + message + severity）
- next_retry_at（如有）
- external_ref（如有）

**实现方式**: 直接调 `SqliteStore`，模式同 `handle_human_review_decide`（用 `_runtime_context()`），不经过 HTTP 层，不依赖 server 运行。

### 1.3 `approve` / `reject`

**用途**: operator 直接在 CLI 接受或拒绝一个 task，不需要开浏览器。

**命令**:
```
apl approve <task_id> [--reason TEXT] [--project-root PATH]
apl reject  <task_id> [--reason TEXT] [--project-root PATH]
```

**行为**:
- `approve`: 把 task status 转换到 `ACCEPTED`，写 audit event
- `reject`: 把 task status 转换到 `REJECTED`，写 audit event
- 成功后打印 audit event_id
- 不经过 HTTP 层，直接调 store + transitions

**前置校验**:
- `approve` 只允许在 `QC`、`HUMAN_REVIEW` 状态调用
- `reject` 只允许在 `HUMAN_REVIEW`、`QC`、`ACCEPTED` 状态调用
- 状态不符时打印明确错误（不 raise 裸异常）

### 1.4 `merge`

**用途**: 把当前 pipeline 的 ACCEPTED task 合并为一个 JSONL 输出，面向 operator 日常操作的简洁接口。

**命令**:
```
apl merge [--pipeline-id ID] [--output PATH] [--project-root PATH]
```

**行为**:
- 调用 `ExportService` 把 ACCEPTED task 的 `annotation_result` artifact 合并写入 JSONL
- 默认输出路径：`<project-root>/exports/merged-<timestamp>.jsonl`
- 打印写入路径和 task 数量

**与 `export training-data` 的关系**: 两者共存。`merge` 是面向 operator 的简洁版（参数少、路径自动），`export training-data` 是面向 ML 工程师的完整版（更多控制选项）。底层共享 `ExportService`。

---

## 线 2：Plugin Protocol

### 2.1 目录结构

```
annotation_pipeline_skill/plugins/
  __init__.py
  base.py            ← Protocol 声明
  registry.py        ← 简单注册表
  jsonl_adapter.py   ← DatasetAdapter 参考实现
```

### 2.2 `base.py` — Protocol 声明

声明三个最核心的扩展接口：

```python
from typing import Protocol, Iterable, runtime_checkable
from annotation_pipeline_skill.core.models import Task, ArtifactRef, SourceRef, TaskDraft

@runtime_checkable
class DatasetAdapter(Protocol):
    def discover_sources(self, config: dict) -> list[SourceRef]: ...
    def build_tasks(self, source: SourceRef, task_size: int) -> Iterable[TaskDraft]: ...
    def build_manifest(self, draft: TaskDraft) -> dict: ...

@runtime_checkable
class Validator(Protocol):
    def validate_output(self, task: Task, artifact_path: str) -> "ValidationResult": ...

@runtime_checkable
class MergeSink(Protocol):
    def write(self, tasks: Iterable[Task], output_path: str) -> "MergeReport": ...
```

`PromptBuilder`、`RepairStrategy`、`QcPolicy` 加 `# TODO: V2` 注释，本次不声明，避免过度设计。

### 2.3 `registry.py` — 最简注册表

```python
_adapters: dict[str, DatasetAdapter] = {}
_validators: dict[str, Validator] = {}
_merge_sinks: dict[str, MergeSink] = {}

def register_adapter(name: str, adapter: DatasetAdapter) -> None: ...
def get_adapter(name: str) -> DatasetAdapter: ...
# 同样模式 for validators, merge_sinks
```

**不使用** `pkg_resources` entry_points 或 importlib 动态加载——就是有类型的 dict。用户在项目入口调 `register_adapter("jsonl", JsonlDatasetAdapter())` 即可。

### 2.4 `jsonl_adapter.py` — 参考实现

把 `cli.py:handle_create_tasks` 里的 JSONL 读取逻辑提取为 `JsonlDatasetAdapter`，显式实现 `DatasetAdapter` Protocol。

CLI handler 改为：
```python
adapter = get_adapter(args.adapter or "jsonl")
tasks = adapter.build_tasks(source, args.task_size)
```

原有行为不变，只是把逻辑从 CLI handler 移入 adapter 类。

### 2.5 `subagent_cycle.py` 分拆——两处提取

#### 提取 1：`runtime/prompt_builder.py`

提取范围：`_annotation_instructions`、`_build_qc_instructions`、`_annotation_prompt`、`_qc_prompt`、`_build_conventions_block`、`_delta_feedback_items`、`_artifact_context`、`_slim_annotation_payload` 等 prompt 构建方法。

这些方法读取状态、返回字符串，无副作用，适合提取为独立类：

```python
class AnnotationPromptBuilder:
    def __init__(self, store: SqliteStore, project_id: str): ...
    def build_annotation_prompt(self, task: Task, ...) -> str: ...
    def build_qc_prompt(self, task: Task, ...) -> str: ...
```

`SubagentRuntime` 持有 `self._prompt_builder`，原调用位置改为委托调用，公共接口不变。

#### 提取 2：`runtime/annotation_validator.py`

提取范围：`_check_annotation_validation`、`_check_verbatim_spans`、`_auto_align_corrected_annotation`、`_verbatim_candidate_spans`、`_record_validation_feedback`。

```python
class AnnotationValidator:
    def validate_output(self, task: Task, payload: Any) -> ValidationResult: ...
    def check_verbatim_spans(self, task: Task, payload: Any) -> list[VerbatimViolation]: ...
```

这个类实现 `plugins/base.py` 中的 `Validator` Protocol，形成文档和代码的第一个真实对应点。

**不拆的部分**: Arbiter 逻辑（状态依赖复杂）、entity statistics（与 QC 流程深度耦合）。两周内重构风险超过收益。

**预期效果**: `subagent_cycle.py` 从 3917 行降至约 2700 行，两个提取模块各约 300–400 行。

---

## 线 3：文档对齐

### 3.1 `TECHNICAL_ARCHITECTURE.md` — 三处修改

**§6.1 状态机**

删除"当前实现使用 7 个 task status"的错误描述，更新为正确的 10 个状态：

| Status | 含义 |
|---|---|
| `draft` | task 已创建，manifest 未生成 |
| `pending` | 等待 worker claim |
| `annotating` | 标注 LLM 调用进行中 |
| `qc` | validation 通过，QC 进行中 |
| `arbitrating` | 仲裁 LLM 调用进行中，或 mechanical retry 等待 pickup |
| `human_review` | 需要人工判断 |
| `accepted` | 终态 — 通过所有检查 |
| `rejected` | 终态 — 人工拒绝 |
| `blocked` | 需要人工干预才能继续 |
| `cancelled` | 已取消 |

补充说明：`validating` 是 inline 步骤（annotation 写完后立刻执行），不是独立 task status；失败后 task 回 `pending` 重试。

**§9 应用服务**

补充以下在代码中存在但文档未提及的服务，每个一行说明职责：

- `coordinator_service` — 多项目协调调度
- `distribution_service` — span/entity 类型分布统计与缓存
- `entity_convention_service` — 高确定性约定读写（注入 prompt 用）
- `entity_statistics_service` — 所有 ACCEPTED 决策的统计分布（prior verifier 用）
- `export_service` — ACCEPTED task 导出为训练数据
- `human_review_service` — HR 阶段的决策写入和修正
- `outbox_dispatch_service` — 外部状态回传和结果提交的可靠队列 drain
- `provider_config_service` — provider profile 读取和校验
- `readiness_service` — 项目就绪度检查
- `row_dedup_service` — 输入行去重检测
- `row_mask_service` — 屏蔽低质量或重复行

**§14.2 API 端点**

更新 API 路径映射表，把文档描述路径对应到实际实现路径：

| 文档描述 | 实际路径 |
|---|---|
| `GET /dashboard` | `GET /api/kanban` + `GET /api/dashboard-stats` |
| `GET /settings` | `GET /api/config` + `GET /api/providers` + `GET /api/annotators` |
| `POST /tasks/<id>/retry` / `approve` / `reject` | `POST /api/tasks/<id>/move` |
| `GET /tasks/<id>/feedback` | 包含在 `GET /api/tasks/<id>` 响应体内 |

补充文档完全未描述的端点：`/api/conventions`、`/api/posterior-audit`、`/api/distribution`、`/api/row-dedup`、`/api/type-statistics`、`/api/entity-statistics`、`/api/typical-text`、`/api/alerts`、`/api/coordinator`、`/api/readiness`、`/api/export-file`、`/api/annotation-rules-document`、`/api/documents`、`/api/jobs/<id>`、`/api/runtime/monitor`。

### 3.2 `PRODUCT_DESIGN.md` — §10.1 补注

在现有状态机列表前加一段说明，不删现有内容：

> **注（当前实现）**: `validating`、`repair_needed`、`merged`、`ready`、`retry_scheduled` 是通用框架的概念阶段，当前实现未作为独立 task status 建模。validation 是 inline 步骤；repair 路径通过 `pending` 重试循环 + `arbitrating` 仲裁实现；merge 通过 ExportService 执行，不改变 task status。`arbitrating` 是当前实现特有的仲裁状态。

产品愿景、功能规划、用户故事部分保持不变——这些是前瞻性内容，不对齐到当前实现。

---

## 不在本 sprint 范围内

- Arbiter 逻辑拆分（风险过高）
- entity statistics 从 subagent_cycle 中解耦
- `PromptBuilder`、`RepairStrategy`、`QcPolicy` Protocol 声明（留 V2）
- API 路径重命名（破坏性变更，需要前端同步修改）
- `templates/` 和 `examples/` 目录创建
- PreviewRenderer 实现（多模态功能）

---

## 验收标准

**线 1**:
- `apl serve` 命令可正常调用，不报 unrecognized command
- `apl inspect <id>` 输出 task status、attempts 和 open feedback，无 HTTP 依赖
- `apl approve <id>` / `apl reject <id>` 写入 audit event，打印 event_id
- `apl merge` 输出合并 JSONL，打印文件路径和 task 数量

**线 2**:
- `plugins/base.py` 中 Protocol 类可被 `isinstance()` 检查（`@runtime_checkable`）
- `JsonlDatasetAdapter` 通过 `isinstance(adapter, DatasetAdapter)` 检查
- `AnnotationValidator` 通过 `isinstance(v, Validator)` 检查
- `subagent_cycle.py` 现有测试全部通过（提取不改变行为）
- 文件行数从 3917 降至 ≤ 2800

**线 3**:
- TECHNICAL_ARCHITECTURE §6.1 状态数量描述正确（10 个）
- TECHNICAL_ARCHITECTURE §9 包含所有 11 个实际存在的 service
- TECHNICAL_ARCHITECTURE §14.2 API 路径与代码一致
- PRODUCT_DESIGN §10.1 有清晰的"当前实现"注释
