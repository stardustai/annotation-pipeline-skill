# Annotation Pipeline Skill 产品设计文档

## 1. 文档目的

本文档定义一个通用开源 skill 的产品设计。这个 skill 不复制 `memory-ner` 当前实现，而是抽象它已经验证过的产品价值：

- 把大批量标注工作拆成可追踪的任务单元
- 让标注、质检、修复、合并形成稳定流水线
- 让人类运营者只处理少量高价值决策，而不是手动 babysit 全流程
- 让不同数据集、不同 schema、不同模型供应商可以复用同一套作业编排方式

建议产品名：

- `annotation-pipeline-skill`


## 2. 背景与问题

### 2.1 现有标注工作的共性痛点

无论是 NER、关系抽取、结构化抽取、分类，还是复杂 JSON 标注，团队通常都会遇到同一组问题：

- 原始数据量大，不能一次性交给单个模型或人工处理
- 标注规则会不断迭代，导致返工和重跑频繁
- 单次标注结果质量不稳定，需要抽样 QC 或规则校验
- 失败任务容易卡死，运营者需要频繁手动重试
- 不同数据源、不同 schema、不同模型工具链之间缺少统一编排方式
- 最终“什么已经做了、为什么失败、该如何修复”缺少可追踪记录

### 2.2 当前工具值得借鉴的产品结论

从 `annotation manager` 里提炼出的有效产品结论是：

- 任务最小单位应该是“可独立重试、可独立审计、可独立合并”的 slice
- 标注不是一步完成，而是 `prepare -> annotate -> validate -> qc -> repair -> accept/reject -> merge`
- 需要把“业务状态”和“运行时执行”明确区分
- 失败恢复必须是系统能力，不应依赖操作员临时判断
- UI 的核心价值不是炫技，而是给运营者一个可信的当前真相和少量必要控制


## 3. 产品定位

`annotation-pipeline-skill` 是一个面向 AI 代理工作流的开源 skill，用于：

- 初始化通用标注流水线
- 根据数据源切分任务
- 编排标注、校验、QC、修复、合并阶段
- 通过插件化接口接入任意 schema、validator、prompt builder、merge sink
- 为运营者提供最小可用的状态查看和控制能力

它不是：

- 一个只服务 NER 的专用工具
- 一个强绑定 Codex/Claude 的私有脚本集
- 一个必须依赖 systemd、Redis、Docker 才能运行的重型平台
- 一个替代 Label Studio 这类逐条人工点击标注 UI 的产品
- 一个 Streamlit 应用


## 4. 目标用户

### 4.1 主要用户

- 数据工程师：批量构建训练数据、评测数据、对齐数据
- Applied AI 工程师：把模型生成结果转成结构化资产
- Agent workflow 开发者：希望把“批处理标注 + QC + 修复”封装为技能
- 小团队运营者：没有专门平台团队，但需要可靠的大规模标注流程

### 4.2 次要用户

- 研究人员：快速搭建实验性数据集流水线
- 开源社区维护者：为不同任务类型提供 adapter 模板


## 5. 核心使用场景

### 5.0 从安装 skill 到实际使用的用户旅程

下面是一条具体 user story，用来约束后续开发时的端到端体验。

#### User Story：数据工程师首次安装并跑通一个标注项目

作为一个数据工程师，我希望把 `annotation-pipeline-skill` 安装到本地 agent 环境后，在一个新项目里快速接入数据源、配置 provider、打开看板、启动流水线，并在任务完成后拿到可追踪的标注结果，这样我不需要从零写任务编排、重试、QC 和状态看板。

##### 1. 发现和安装 skill

用户先在 agent 环境里安装 skill：

```bash
codex skill install annotation-pipeline-skill
```

或从本地路径安装：

```bash
codex skill install /path/to/annotation-pipeline-skill
```

安装完成后，用户可以查看 skill 暴露的入口：

```bash
codex skill list
annotation-pipeline --help
```

系统应该清楚显示：

- 当前 skill 版本
- 可用命令
- 初始化项目的下一步命令
- 本地依赖是否缺失，例如 Python 包、Node/TypeScript 前端依赖

##### 2. 初始化一个标注项目

用户进入自己的工作目录并初始化项目：

```bash
mkdir contract-annotation-demo
cd contract-annotation-demo
annotation-pipeline init
```

系统生成最小项目结构：

```text
.annotation-pipeline/
  project.yaml
  pipeline.yaml
  providers.yaml
  stage_routes.yaml
  external_tasks.yaml
  tasks/
  attempts/
  events/
  artifacts/
  runtime/
  snapshots/
adapters/
rules/
schemas/
web/
```

初始化结束时，系统输出一个可执行 checklist：

- 配置数据来源：本地 JSONL 或外部任务 API
- 配置 schema / validator
- 配置 provider registry 和 stage routes
- 运行 `annotation-pipeline doctor`
- 启动 API 和 TypeScript 看板

##### 3. 选择任务来源

用户可以选择两种入口之一。

本地文件模式：

```bash
annotation-pipeline create-tasks \
  --source data/raw/input.jsonl \
  --task-size 500 \
  --adapter jsonl_basic
```

外部任务 API 模式：

```bash
annotation-pipeline external configure \
  --adapter http_json \
  --pull-url https://tasks.example.com/api/tasks/pull \
  --status-url https://tasks.example.com/api/tasks/status \
  --submit-url https://tasks.example.com/api/tasks/submit \
  --auth-secret-ref env:EXTERNAL_TASK_API_TOKEN

annotation-pipeline external pull --limit 100
```

系统行为要求：

- 本地文件模式会创建内部 `Task`、raw slice artifact 和 manifest。
- 外部 API 模式会创建带 `ExternalTaskRef` 的内部 `Task`。
- 重复拉取同一个外部 task 必须幂等，不产生重复 task。
- 每个创建动作都写 audit event。

##### 4. 配置 LLM provider 和阶段路由

用户配置 provider registry：

```bash
annotation-pipeline providers add general_llm \
  --kind openai_compatible \
  --models general-large,general-small \
  --default-model general-large \
  --secret-ref env:GENERAL_LLM_API_KEY

annotation-pipeline providers add review_llm \
  --kind chat_completion \
  --models review-large,review-fast \
  --default-model review-large \
  --secret-ref env:REVIEW_LLM_API_KEY
```

然后配置每个阶段的 provider route：

```bash
annotation-pipeline settings set-route \
  --stage annotation \
  --primary-provider general_llm \
  --primary-model general-large \
  --primary-effort medium \
  --fallback-provider review_llm \
  --fallback-model review-fast \
  --fallback-effort high

annotation-pipeline settings set-route \
  --stage qc \
  --primary-provider review_llm \
  --primary-model review-large \
  --primary-effort high \
  --fallback-provider general_llm \
  --fallback-model general-large \
  --fallback-effort high
```

系统行为要求：

- provider secret 只保存引用，不保存明文。
- settings 校验 provider id、model、effort 是否可用。
- 配置变更写 settings audit event。
- 配置变更只影响后续 dispatch，不改写已经运行中的 attempt。

##### 5. 运行 doctor 检查

用户运行：

```bash
annotation-pipeline doctor
```

系统检查：

- project config 是否完整
- task store 是否可读写
- validator 是否可执行
- provider route 是否有效
- secret reference 是否存在
- TypeScript 前端依赖是否安装
- 外部任务 API 是否可达，如果启用

doctor 的输出必须是行动导向的，告诉用户下一步具体该改哪个配置或运行哪个命令。

##### 6. 启动 API、runtime 和 TypeScript 看板

用户启动本地服务：

```bash
annotation-pipeline api serve
annotation-pipeline run --runtime local_subprocess
annotation-pipeline dashboard serve
```

`dashboard serve` 启动 TypeScript Web 看板。看板第一屏显示：

- runtime health 和 heartbeat age
- task counts by status
- live workers / max concurrency
- queued counts by stage
- due retry 和 stale active task 摘要
- provider route 配置入口
- task 状态分栏和过滤搜索

看板不直接读取本地 task 文件。它只消费 `GET /dashboard`、`GET /tasks/<task_id>`、`GET /settings` 和控制 API。

##### 7. 启动流水线并观察推进

用户可以通过 CLI 启动：

```bash
annotation-pipeline run --once
```

或在看板中开启自动派发：

```text
Auto dispatch: on
Max concurrent tasks: 8
Max starts per cycle: 4
```

系统开始推进任务：

```text
ready -> annotating -> validating -> qc -> accepted -> merged
```

如果 deterministic validation 失败：

```text
validating -> repair_needed
```

如果 provider 超时、限流或不可用：

```text
attempt failed -> retry_scheduled
```

如果外部任务 API 启用，系统在关键阶段写入 outbox 并异步回传状态。

##### 8. 处理异常和人工干预

当看板显示异常时，用户可以：

- 查看 task detail，读取 attempts、events、artifacts、provider route 和 external ref
- 查看 QC feedback records，理解每条失败意见对应的 source line、output span、failure code 和建议动作
- 对单个 task 执行 retry、approve、reject、merge
- 对 repair_needed task 选择修复方式：bulk code repair、annotator rerun、manual annotation
- 暂停某个阶段的 provider route，并切换 fallback route
- 查看 external outbox 或 dead-letter，重新 drain 外部状态回传

所有人工动作都必须：

- 通过 application service 执行
- 写 audit event
- 返回 audit event id
- 在看板刷新后可追踪

##### 9. 获取结果和收尾

本地文件模式下，用户运行：

```bash
annotation-pipeline merge --all-accepted
```

外部任务 API 模式下，系统对 accepted / rejected / merged task 调用 submit/status API。

最终用户可以导出：

- merged artifact
- validation report
- QC report
- task audit history
- external submission report

##### 10. 首次成功的验收标准

这条 user story 的成功标准是：

- 用户能在一个新目录中完成 `init -> configure -> doctor -> create/pull tasks -> run -> inspect dashboard -> merge/submit`。
- 至少一个 task 完整经过 annotate、validate、QC、accept、merge。
- 看板能显示真实 runtime health、worker counts、task detail 和 provider route。
- provider secret 没有出现在 task state、artifact、dashboard snapshot 或前端 state 中。
- 外部 API 模式下，重复 pull 不创建重复 task，状态回传失败会进入 outbox 并可重试。


### 场景 A：从原始 JSONL 构建结构化标注集

用户提供：

- 原始数据文件
- 标注 schema
- 标注规则
- validator

skill 完成：

- 切分 task
- 调度 annotate
- 运行 deterministic validate
- 抽样 QC
- 失败重试或进入修复
- 合并通过结果

### 场景 B：对既有标注集进行规则升级

用户更新规则或 validator 后，skill 能够：

- 找出受影响 task
- 标记需要 rerun 或 repair 的 task
- 保留历史 attempts
- 在不丢失 trace 的前提下重新推进流水线

### 场景 C：多模型协作标注

用户可以配置：

- annotation provider
- QC provider
- fallback provider

skill 负责统一生命周期管理，而不是把 provider 逻辑散落到业务代码中。

### 场景 D：人机协同审核

当 deterministic validator 或 QC 不能直接放行时，skill 提供：

- rejection reason
- retry guidance
- artifact trace
- human review hook
- QC feedback record，用于把质检意见结构化保存下来
- annotator repair context，让下一轮 annotator 能基于反馈修改输出
- repair mode decision：由 annotator 或 repair strategy 判断是批量代码修复，还是进入人工标注修复

### 场景 E：外部任务平台接入

用户已有一个外部任务系统，负责生产 task、接收结果、追踪 task 状态。skill 需要作为本地或服务端编排层：

- 从外部 API 拉取待处理 task
- 把外部 task 映射成框架内部 `Task`
- 在每个关键阶段向外部系统回传状态
- 在 accepted / rejected / merged 后提交结果或失败原因
- 在外部 API 暂时不可用时保留本地审计记录和重试计划

MVP 外部任务 API 采用 pull + status callback 模式：

- skill 主动 pull task，不接收 webhook。
- 每个关键阶段通过 status callback 回传。
- 最终结果通过 submit endpoint 提交。
- status 和 submit 都先写入 outbox，再由 drain 可靠发送。

外部平台接入必须通过 adapter 完成，不能把某个 API 协议写进 core。Webhook 接收、外部主动推送和公网回调服务不进入 MVP。

### 场景 F：运营者通过看板配置与干预流水线

运营者需要一个可视化看板来完成日常管理：

- 查看 scheduler/runtime 健康状态、heartbeat、队列和 live worker 真相
- 按 source、status、task id 搜索和过滤任务
- 查看每个 task 的阶段、runtime lease、重试时间、失败摘要、QC 历史和 artifact trace
- 启动、停止、重试、接受、拒绝或合并少量任务
- 配置全局并发、每周期启动上限、自动派发开关
- 配置 annotation、QC、repair、merge 等阶段的默认 provider、fallback provider、model 和 effort

参考 `memory-ner/annotation manager/annotation_manager_streamlit.py` 的信息架构和运营行为，但新 skill 不使用 Streamlit。看板应使用 TypeScript Web 框架实现，并建立在稳定的 dashboard API 和 settings API 上。

### 场景 G：按能力选择标注员并扩展到多模态标注

不同 task 可能需要不同能力的 annotator。系统需要支持将标注员、模型工具或外部检测服务登记为 `AnnotatorProfile`，并按能力选择合适的执行者：

- 文本抽取 annotator：适合 JSONL、文档、对话等文本任务
- 视觉检测 annotator：适合图片 bounding box、mask、keypoint 等任务
- 视频 annotator：适合帧级检测、片段标注、轨迹标注
- 点云 annotator：适合 3D box、instance segmentation、轨迹或空间属性标注
- 人类 annotator：适合需要人工判断、纠错或最终审核的任务

具体例子：

1. 用户导入一批图片 task，并配置视觉检测 annotator。
2. annotator 调用一个 VC detection 模型，生成 bounding box artifact。
3. 系统把原图和 bounding box overlay 渲染成 preview artifact。
4. task 先进入 QC；QC 可以使用 bbox JSON、原图和 preview artifact 做检查。
5. 如果该 pipeline 配置了 Human Review，QC 之后进入 Human Review。
6. TypeScript 看板展示带 bounding box 的图片，reviewer 检查标注效果。
7. 如果效果可接受，reviewer 接受 task；如果不符合要求，reviewer 生成 feedback record，选择重新调用检测模型、批量代码修复 box 格式，或进入人工标注。

这个能力必须通过 adapter 和 capability registry 扩展，不能把图片、视频或点云规则写死进 core。


## 6. 产品目标

### 6.1 核心目标

- 让用户在 30 分钟内搭起一个可运行的标注流水线
- 让每个 task 都具备完整 traceability
- 让失败任务自动恢复，不因单点故障拖垮全局
- 让 skill 支持任意任务类型，而不是写死 NER
- 让“框架层”和“项目私有标注逻辑”完全分离

### 6.2 非目标

- 不提供复杂人工逐 token 标注界面
- 不负责训练模型本身
- 不把所有数据平台能力都纳入第一版
- 不做强耦合云服务
- 不在 core 中内置某个外部任务平台的私有 API
- 不把 provider routing 绑定到 Codex、Claude 或任何单一模型供应商


## 7. 产品原则

### 7.1 Trace First

每个 task 都必须回答：

- 输入是什么
- 当前在哪个阶段
- 失败过几次
- 为什么失败
- 产物在哪
- 下一步是什么

### 7.2 Deterministic Before Model

凡是可以 deterministic 做的事情，优先 deterministic：

- schema 校验
- 行数对齐
- manifest 一致性
- 基础 lint
- 合并前 gate

模型用于语义工作，不用于替代基础工程约束。

### 7.3 Framework Core Must Stay Generic

框架核心只关心：

- task 生命周期
- scheduler
- runtime
- artifact orchestration
- plugin contracts

数据集特定规则必须进入 adapter 层。

### 7.4 Operator Burden Must Be Explicitly Reduced

如果某个设计需要运营者频繁盯盘、手动重启、手动判断状态，那就是失败的产品设计。


## 8. 功能范围

### 8.1 MVP 功能

- 初始化 pipeline 项目
- 注册 dataset adapter
- 原始数据切片与 task 生成
- Task 状态机
- Annotation worker 调用
- Deterministic validate 阶段
- QC 抽样与 verdict
- Retry 与 repair hooks
- Accept / reject / merge
- CLI 操作入口
- 文件型 state store
- 最小 dashboard API 和 dashboard snapshot
- TypeScript Kanban 看板，至少支持状态总览、阶段列、任务详情抽屉、过滤搜索和只读 runtime 健康视图
- settings API，支持读取并校验并发、自动派发和阶段级 provider route 配置
- provider registry，支持为 annotation、QC、repair、merge 分别指定 primary 与 fallback provider
- annotator capability registry，支持按 modality、annotation type 和执行方式选择 annotator

### 8.2 V0.2 功能

- 外部任务 API adapter contract
- pull-only task ingestion：从外部 API 拉取任务并转换为内部 task
- status callback outbox：关键阶段状态写入 outbox 并可靠回传
- submit result outbox：accepted / rejected / merged 后提交结果或失败原因
- webhook ingestion 不进入 MVP
- 看板中的任务控制动作：start、stop、retry、approve、reject、merge
- runtime overlay 刷新：看板显示的 live worker 必须来自 runtime truth，不允许只依赖缓存 snapshot
- provider fallback pause / resume：某阶段 provider 不可用时可以暂停该 route 并切到 fallback
- 图片标注预览：支持 image artifact + bounding box overlay 的 read-only preview
- optional Human Review：QC 后可配置人工复核，支持图片 bbox overlay 等多模态证据检查
- Human Review 决策流：reviewer 可在 QC 后执行 accept、reject、request repair，所有动作写 audit event

### 8.3 V1.1 功能

- 可切换 runtime backend
- Provider abstraction
- 更完整的 TypeScript Kanban dashboard
- Human review queue
- Metrics summary
- Rule/version drift detection
- Provider 使用统计、失败率和 fallback 触发原因汇总
- 外部任务 API 的批量 reconcile 和 dead-letter queue
- 多模态 adapter 扩展：video frame sampling、point cloud asset reference、overlay renderer
- annotator capability metrics：不同 annotator 在不同 modality/task type 上的通过率、返修率和平均耗时

### 8.4 后续功能

- 多租户项目空间
- 远程队列后端
- 协作审批流
- 数据集 diff 和增量回放

### 8.5 V1.2 功能 — 经验先验质量保障

针对多 agent LLM 的相关错误 / 信息级联问题，引入"项目内统计先验"作为外
部 verifier。设计文档见
`docs/superpowers/specs/2026-05-17-prior-driven-verifier-design.md`。

核心:
- 两张表分工
  - `entity_statistics`: 项目内所有 ACCEPTED 决策的累计分布（含 arbiter
    决策），verifier 用。HR 决策 5x 权重
  - `entity_conventions` (已存在): 注入 prompt 的高确定性子集，**不含**
    arbiter 决策（避免 cascade）
- Verifier 在三处触发: QC pass / arbiter ruling / HR submit_correction。
  发现决策与项目历史分布显著偏离（样本 ≥ 10 且主类型 ≥ 80%）时升级
- 偏离时调第二个 arbiter（不同 model family）独立判断，两 arbiter 一致
  → 接受；不一致 → HR
- 提供 Posterior Audit tab，operator 手动 Check 后列出所有 ACCEPTED 中
  与当前 stats 不一致的 (task, span, type)，可一键打回 HR

理论依据：Condorcet's jury theorem 要求投票者独立；LLM 之间共享 training
bias 导致 r=0.78 错误相关性，纯 LLM 投票会放大错误。引入项目内经验分布
作为外部 verifier 是文献一致推荐的做法。


### 8.6 V1.3 功能 — Annotation Knowledge Base（按需检索的项目经验）

V1.2 的 entity_conventions 解决了"高确定性约定如何注入到 prompt"，但留下三个
产品级缺口，V1.3 针对这三点补齐。设计文档见
`docs/superpowers/specs/2026-05-19-annotation-knowledge-base-design.md`。

#### 产品动机（三个缺口）

1. **Prompt 注入是被动的，token 经济学差**。`find_matches_in_text` 把所有
   命中的 convention 塞进 prompt，无论这一行 row 实际需不需要。项目跑到
   后期 convention 表上千条时，prompt 的"已知知识"段会挤掉真正的 row
   文本，agent 反而看不清要标注什么。
2. **统计量对 agent 不可执行**。告诉 agent "这个 span 的 type_entropy
   是 0.85" 等价于告诉它"这是个有争议的 span"——但没告诉它"在什么上下文
   下应该选哪个 type"。LLM 擅长 mode-match 具体例子，不擅长根据熵值反推
   决策。
3. **Agent 无法主动追问历史**。遇到一个不认识的 span，agent 只能用
   training 里的常识猜，没法说"等一下，让我先查查这个项目以前怎么标
   过的"。memory-ner 项目里 1300 行的 `ANNOTATION_GUIDE.merge_updates.md`
   就是人工填补这个缺口的产物——我们要的是同样的效果，但自动化、可追溯、
   不靠人维护。

#### 核心（产品层面）

- **新的 agent 工具：`check_past_experience(entry)`**。MCP 协议层暴露给
  annotator / QC / arbiter，agent 主动决定何时调用。Token 成本只在 agent
  实际"困惑"时支付，不再随 convention 表线性增长。
- **返回的是案例，不是数据**。每次调用返回当前 convention、type
  distribution、**每个 type 配最多 3 条来自不同上下文的真实 row 片段**
  （MinHash farthest-first 选出最不相似的样本）、以及 wordfreq Zipf
  分数。Agent 看到 "Apple → organization 的 3 个例子都是 customer
  support 语境；Apple → product 的 3 个例子都在讲 iPad 硬件" 就能直接
  类比当前 row 决定。
- **零新表，全部由现有 entity_conventions + posterior_audit 组合出来**。
  唯一 schema 变化：`entity_conventions.proposals_json` 里每条 proposal
  新增 `row_id` + `context_snippet` 字段（一个 ±80 字符的 span 周围窗口）。
  无迁移、无回填窗口、无写路径分裂。
- **CJK 升级附带价值**。`similarity.shingle()` 之前对中文整行 hash（退化
  为单 shingle），既影响 KB 的多样性采样，也使 row_dedup 对中文 row 实际
  上无效。新增 jieba CJK gate 修复两者，ASCII 路径完全不变。
- **provider switch 不污染主 shell**。`isolated_claude_home` 把 profile
  的 `base_url` 注入子进程 env，operator shell 里的 `ANTHROPIC_BASE_URL`
  永远不变。一个 annotator 子任务跑 DeepSeek 不会让 operator 同时开着的
  Claude Code 也切去 DeepSeek。
- **系统级 prompt，不是项目级 rule**。指导 agent 何时调用工具的文字写在
  runtime 的 `_annotation_instructions()` 里（含一段 conditional
  paragraph：tool 没注册时段落空 no-op），不要求每个项目自己往
  `annotation_rules.yaml` 里抄一份。

#### 与 V1.2 的分工

V1.2 与 V1.3 是互补的两层：

- **V1.2（entity_statistics + posterior audit）**：被动 verifier。Agent
  做完决策、QC pass 之后，runtime 主动比对项目分布、发现偏离时升级到
  二号 arbiter / HR。Agent 不参与这层逻辑。
- **V1.3（check_past_experience）**：主动咨询。Agent 在标注前/标注中
  自己决定要不要查 KB。让"经验先验"从事后审计的工具，扩展为事前/事中
  的决策辅助。

两者写入的是同一个 `entity_conventions` 表——V1.2 的 verifier-time 决策
和 V1.3 的 agent-time 查询共享同一份真理。

#### 取舍

被刻意留在 V2 / 拒绝掉的能力：

- **Row-level BM25 / 语义检索**。Brainstorm 时讨论过给 KB 加一个
  "找类似 row" 的能力。否决：per-span 检索已经覆盖了"agent 心里有具体
  候选 span"这个主要 case；row-level 索引引入 jieba 索引 + 重建调度 +
  freshness 不变量的运维负担；目前没有数据支持 per-span 不够用。等真有
  信号了再开。
- **新的 SpanKnowledge 表 + 后台写入**。否决理由是写路径分裂会引入
  denormalize bug 和 backfill 窗口里的 staleness；现有数据 + 查询时组合
  才是更稳的设计。
- **自动合成 annotation_rules.yaml**。memory-ner 的人工 merge guide 是
  范例不是要复刻——KB 用案例驱动学习取代显式规则，避免运营者维护一份
  会过期的"知识"文件。

实测验证：在 disputed 的 Apple span 上，agent 看了"organization 例子都
是 customer support 语境"后，正确把当前 row 的 "Apple's customer support
helped me…" 标为 organization；同一行的 Android 因为 active convention
明确，直接复用，不再走 KB 查询。Token 路径精确按需付费。


## 9. 信息架构

产品层面包含十个对象：

- `Project`
  - 一个标注工程，定义规则、schema、adapter、runtime 配置
- `Task`
  - 一个独立 slice，是最小调度和追踪单位
- `Attempt`
  - 一次具体执行记录，包括产物、失败原因和修复上下文
- `Artifact`
  - 原始 slice、标注结果、QC 报告、validator 输出、merge report 等
- `FeedbackRecord`
  - 一条可追踪的质检或校验反馈，包含失败类型、定位信息、建议动作和修复决策
- `Run`
  - 某个 runtime/backend 上的一次执行实例
- `ProviderRoute`
  - 某个阶段的 primary/fallback provider、model、effort 和 pause 状态
- `AnnotatorProfile`
  - 一个可被调度的标注员、模型工具或人工队列，声明 modality、annotation type、输入输出 artifact 和执行方式
- `MediaPreview`
  - 多模态标注的可视化检查产物，例如带 bounding box 的图片、视频帧 overlay 或点云 viewer 状态
- `ExternalTaskRef`
  - 外部任务系统中的 task 引用、幂等键、状态回传和结果提交记录


## 10. 核心工作流

### 10.1 生命周期

建议通用状态机：

> **当前实现说明**（与通用状态机的偏差）：
> - `validating` 是 annotation worker 内部的 inline 步骤，不是独立 task status；失败后 task 回 `pending` 并写 BLOCKING FeedbackRecord。
> - `repair_needed` 在当前实现中通过 `pending` 重试循环 + `arbitrating` 仲裁状态覆盖，没有独立状态。
> - `ready` 等同于 `pending`（draft 写完即直接变 pending）。
> - `merged` 通过 `ExportService` 实现为操作而非状态；accepted task 的 status 不改变。
> - `retry_scheduled` 由 `next_retry_at` 字段在 task metadata 里表达，不是独立状态。
> - `arbitrating` 是当前实现特有的仲裁状态，对应通用模型中"进入 repair/arbiter 判断"这一阶段。
>
> 上述偏差反映了当前 NER 项目驱动的实现选择；通用框架扩展时可以重新建模这些为独立状态。

`draft -> ready -> annotating -> validating -> qc -> human_review -> accepted/rejected/repair_needed -> merged`

补充状态：

- `blocked`
- `retry_scheduled`
- `cancelled`

### 10.2 Task 推进规则

- `draft -> ready`
  - task 切片完成且 manifest 生成成功
- `ready -> annotating`
  - scheduler 成功分配执行资源
- `annotating -> validating`
  - worker 提交结果产物
- `validating -> qc`
  - deterministic gate 通过
- `validating -> repair_needed`
  - deterministic gate 失败
- `qc -> accepted`
  - QC 通过阈值，且 Human Review policy 未要求复核
- `qc -> human_review`
  - QC 通过阈值，且 pipeline 强制复核或 QC policy 判定该 task 风险较高
- `qc -> repair_needed`
  - QC 不通过但允许修复，同时生成 `FeedbackRecord`
- `human_review -> accepted`
  - reviewer 接受 QC 后结果
- `human_review -> rejected`
  - reviewer 拒绝结果
- `human_review -> repair_needed`
  - reviewer 要求修复，并生成或更新 `FeedbackRecord`
- `repair_needed -> annotating`
  - annotator 基于 feedback record 重新跑标注
- `repair_needed -> validating`
  - repair strategy 执行批量代码修复后重新进入 deterministic validation
- `repair_needed -> blocked`
  - feedback 指向需要人工标注或人工决策的问题
- `accepted -> merged`
  - merge sink 成功写入目标资产

### 10.2.1 Human Review 触发策略

Human Review 使用混合触发策略：

- Pipeline-level policy 可以强制所有 QC 通过的 task 进入 Human Review。
- QC policy 可以按风险把单个 task 送入 Human Review。
- 两者任一命中时，task 从 `qc` 进入 `human_review`。
- 两者都未命中时，QC 通过的 task 直接进入 `accepted`。

风险路由可以基于结构化 QC 输出，例如低置信度、边界案例、抽样命中、多模态 preview evidence、历史失败次数或 annotator 质量指标。不要基于 task 文本关键词做硬编码路由。

### 10.3 Kanban 默认列

MVP 默认采用 operational Kanban columns：

- `Ready`
- `Annotating`
- `Validating`
- `QC`
- `Human Review`
- `Repair`
- `Accepted`
- `Rejected`
- `Merged`

每列对应一个或一组明确 task status。过滤搜索只改变卡片集合，不改变列结构。


## 11. 用户体验设计

### 11.1 CLI 优先

MVP 阶段以 CLI 为主，理由：

- skill 的第一用户是工程师
- 便于集成到 agent 工作流
- 更适合开源传播和自动化

关键命令：

- `init`
- `create-tasks`
- `run`
- `retry`
- `inspect`
- `approve`
- `merge`
- `doctor`

### 11.2 Dashboard 是运营主界面

Dashboard 不追求复杂人工标注，但它是运营主界面。它至少需要满足：

- 全局状态总览
- scheduler/runtime 健康、heartbeat age、service state
- 队列数量和 live worker 数量，按 annotation、QC、repair、merge 分解
- 各阶段 task 数量和 Kanban 阶段列
- 当前运行中的 task、stale active task、due retry task
- 最近失败原因、下一次 retry 时间、runtime lease 信息
- task 详情抽屉，显示 audit event、attempt、artifact、provider route 和外部 task ref
- feedback 面板，显示 validation/QC 反馈历史、失败代码、定位、建议动作和当前 repair decision
- Human Review 面板，显示图片 bounding box、视频帧 overlay 或点云预览，并支持 accept、reject、request repair
- annotator selector，显示可用 annotator capability、当前选择、fallback 和最近质量指标
- source/status/task id 过滤搜索，过滤后仍保持 Kanban 列结构
- start/stop/retry/approve/reject/merge 等少量控制动作
- 可复制的 CLI 命令或 API 请求

Dashboard 的设计原则：

- Dashboard API 返回的是 read model，不直接暴露可变 domain object。
- 看板显示的 worker 真相必须能刷新 runtime overlay；缓存 snapshot 不能掩盖真实 worker 数为 0 的问题。
- 控制动作必须写 audit event，并通过 application service 执行状态转换。
- 设置保存后只影响后续调度和新 attempt，不改写已经运行中的 worker。
- 前端用 TypeScript 框架实现，推荐 React 系生态；前端只消费 HTTP API，不直接读取本地 task store。
- UI 组件和 API payload 类型需要在前端显式建模，避免以未校验的任意 JSON 驱动关键控制动作。

### 11.3 Provider 配置界面

MVP 中 provider 和 annotator 配置以 YAML 为 canonical source。看板需要展示和校验阶段级 provider 配置，但不直接写配置文件：

- 全局 provider registry：provider id、类型、可选模型、默认 effort、凭据引用、启用状态
- 阶段 route：`annotation`、`validate`、`qc`、`repair`、`merge`
- 每个阶段配置 primary route 和 fallback route
- 每条 route 包含 provider id、model、effort、rate limit 标签和 fallback delay
- 支持测试 provider connectivity 和 route validity
- 配置变更通过 CLI 或手动编辑 YAML 完成；UI 只读展示、校验和测试

Provider token 或 secret 只允许保存引用，不允许写入 task、dashboard snapshot 或 artifact。

### 11.3.1 Annotator 能力配置

Annotator 配置和 provider 配置相关，但不是同一个概念：

- `Provider` 描述如何调用模型或外部服务。
- `AnnotatorProfile` 描述某个可调度标注员能处理什么任务、产出什么 artifact，以及是否需要为 QC/Human Review 生成 preview artifact。

一个 annotator profile 至少包含：

- annotator id 和显示名
- 支持的 modality：text、image、video、point_cloud
- 支持的 annotation type：classification、extraction、bounding_box、segmentation、keypoint、track、3d_box 等
- 输入 artifact 类型和输出 artifact 类型
- 使用的 provider route 或 external tool adapter
- preview renderer：例如 image bbox renderer、video frame overlay renderer、point cloud viewer
- human review policy：QC 后是否需要人工复核，以及哪些证据 artifact 必须展示
- fallback annotator 或人工队列

调度器选择 annotator 时应基于 task manifest 中的 modality 和 annotation requirements，以及 annotator profile 中声明的能力。不要用 task 文本里的硬编码关键词来做语义路由。

MVP 中 annotator profile 由 `annotators.yaml` 管理。UI 可以展示 capability match、fallback 和质量指标，也可以触发 validate/test，但不负责新增或编辑 annotator profile。

### 11.4 人工干预点

只在高价值节点允许人工介入：

- 修改 pipeline config
- 标记任务接受/拒绝
- 处理无法自动修复的问题
- 审核 merge 前摘要
- 暂停或恢复某阶段 provider route
- 处理外部任务 API dead-letter 或状态回传冲突
- 对 feedback record 标记修复策略：批量代码修复、annotator 重新标注、人工标注、拒绝

### 11.5 QC Feedback 与修复决策

QC feedback 不是临时 prompt 文本，而是 task trace 的一部分。每次 validation 或 QC 失败都应生成结构化 `FeedbackRecord`：

- 来源：validator、QC provider、human reviewer 或 merge gate
- 定位：source line、output line、span、entity id、artifact ref
- 类型：schema issue、missing item、hallucinated span、low confidence、policy violation、merge blocker 等
- 严重程度：info、warning、error、blocker
- 建议动作：rerun annotation、bulk code repair、manual annotation、reject
- 状态：open、applied、dismissed、superseded

Annotator 下一轮执行时必须能读取 compact feedback bundle，明确知道上一轮失败在哪里、为什么失败、应该怎么改。

Repair decision 分三类：

- `bulk_code_repair`
  - 适合格式错误、重复字段、可确定转换、批量规范化等 deterministic 修复。
  - 修复后回到 `validating`，不需要重新调用 annotator。
- `annotator_rerun`
  - 适合语义缺失、抽取错误、证据不足、需要模型重新理解上下文的问题。
  - rerun prompt 必须包含 compact feedback bundle。
- `manual_annotation`
  - 适合反馈无法自动判定、规则冲突、需要人类裁决或修复成本高于人工处理的问题。
  - task 进入人工队列或 `blocked`，并保留全部 feedback record。

默认策略应由 `RepairStrategy` 给出建议，但 operator 可以在看板上 override。所有 override 都必须写 audit event。


## 12. 配置模型

用户配置应分四层：

- `core config`
  - project root
  - task size
  - runtime backend
  - concurrency
- `task config`
  - dataset adapter
  - schema
  - validator
  - qc policy
- `provider config`
  - annotation provider
  - validation provider（当 deterministic validator 不足以覆盖语义校验时）
  - qc provider
  - repair provider
  - merge provider
  - fallback provider
- `external task api config`
  - pull endpoint
  - submit endpoint
  - status callback endpoint
  - auth secret reference
  - idempotency key strategy
  - retry and dead-letter policy

避免把业务规则混在 runtime 配置里。


## 13. 插件与扩展策略

skill 的扩展点应明确暴露：

- `DatasetAdapter`
- `PromptBuilder`
- `Validator`
- `QcPolicy`
- `RepairStrategy`
- `MergeSink`
- `RuntimeBackend`
- `ProviderClient`

开源仓库需要自带：

- 通用 JSONL adapter
- 抽样 QC policy
- 文件系统 store
- local subprocess runtime
- demo validator


## 14. 成功指标

### 14.1 产品指标

- 首次成功运行时间
- 单个 task 平均推进时长
- 自动恢复成功率
- 需要人工干预的任务比例
- merge 前被 deterministic gate 拦截的比例

### 14.2 工程指标

- Task 状态不一致率
- 丢失 artifact 比例
- worker 异常后恢复时长
- adapter 开发接入时间


## 15. 开源策略

### 15.1 仓库应该包含

- 清晰的核心架构
- 一个最小 demo 项目
- 接口文档
- 本地可运行示例
- adapter 模板
- 测试样例和 fixture

### 15.2 仓库不应该包含

- 私有数据
- 私有 prompt 资产
- 强绑定内部路径
- 强依赖特定个人机器环境的脚本


## 16. 风险与取舍

### 风险 1：过早做成大平台

后果：

- 抽象过重
- 学习成本高
- 开源用户无法快速用起来

应对：

- 保持 MVP 极简
- 先用文件系统 store + 本地 runtime 跑通

### 风险 2：抽象不足，仍是项目脚本集

后果：

- 只能复用一部分命令
- adapter 和 core 混杂

应对：

- 所有业务特定逻辑必须通过接口接入

### 风险 3：把 provider 当成核心产品

后果：

- 被模型工具链变化绑架

应对：

- provider 只做可替换 client
- 产品核心是 pipeline orchestration


## 17. 版本建议

### v0.1

- CLI
- 文件型 task store
- local subprocess runtime
- 通用 JSONL adapter
- deterministic validate + QC

### v0.2

- Web dashboard
- runtime backend abstraction
- retry policy abstraction
- merge sink abstraction

### v0.3

- remote queue backend
- review workflow
- richer metrics and audit tools


## 18. 最终产品定义

`annotation-pipeline-skill` 应该被定义为：

一个开源的、面向 AI 代理与数据工程协作的通用标注流水线 skill。它用统一任务模型和可插拔架构，把大规模数据标注从“零散脚本 + 人工盯盘”提升为“可追踪、可恢复、可扩展”的工程系统。
