# 先规划、按知识需求使用记忆

可选流程 `workflow=need_first`。历史结果及旧配置不改写。默认 `legacy` 保留旧执行方式；新实验使用独立配置和数据库，不能把修改后的源码接到旧实验检查点继续运行。

## 50题实验前复核：need-first/3

2026-10-21补充修复：相同知识的身份哈希不再包含原题ID与仅描述旧病例的conditions；来源版本、任务域、需求goal、引句及其他规则字段仍参与身份。仅need-memory-v1且明确标为origin_context_only的证据适用此规则，不合并不同知识需求或不同引句。独立memory_origin表保存每次原任务、条件和trace；重复提交同一trace与上下文幂等，不自动延长旧记忆有效期。不同goal措辞仍可能形成多条记录，这是保守去重的限制。

最终reused_memory_ids、memory_funnel.cited和memory_supported_needs按已接纳记忆的原文span与真实、可见的最终引句匹配。要求一方完整包含另一方，来源哈希一致；另一处新引句不会仅因来源ID相同被归给旧记忆。source_level_memory_ids另存来源级关联，不能当作片段复用。片段引用仍不证明因果收益或医学正确性。

离线292项回归通过；10轮同知识不同题号探针只保存1条知识（原来10条），9轮零检索；同源新引句误归因探针现为0条复用、0个记忆支持需求。新配置为configs/medical-related-50-v2/compare-api-need-first.json，50题×4条件；同等输出上限2600，避免复杂需求审核挤占旧1800上限。具体API结果以新结果目录记录为准。

## 真实响应兼容复核：need-first/4

最初API检查批次完成10题后停止：有4题Investigator将已resolve的mem-ID写入evidence_ids。28次已保存调用、53872个已知Token，无未知请求；该批次单独保存在medical50-api-fourway-need-first-v2，不计入正式对照。

调查解析现可将已成功回读的记忆ID映射为对应真实来源，仍要求来源在当前evidence_pool可见；未知记忆、未回读记忆及不可见来源不能映射。映射事件独立记录。4份原始失败响应离线重放均通过，不调用API。提示明确记忆ID与来源ID分工，并补全Reasoner的支持程度、条件判断枚举，避免中文自由标签或not_applicable等状态误入审核。

正式批次使用独立目录medical50-api-fourway-need-first-v3，旧批次保留，不在改变源码后续接旧检查点。最终离线294项回归通过。

## 同源多片段修复：need-first/2

需求驱动流程现已分开来源读取预算与记忆片段预算：

- 每次按需求召回可保留同一来源下的不同记忆ID，旧流程的来源级召回去重保持原样。
- 跨需求轮流选择，默认最多3个唯一来源、9条记忆引用；同源新增片段不再占用第二个来源读取名额。
- 每个来源版本在一题内只读取和校验一次，成功和失败结果都缓存；返回调查时即使来源读取余额为0，也能解析该来源的其他片段。
- 所有选中引句合并保留，不再按来源截成前三条。先按记忆引句构建带邻近上下文的视图，再检查上下文预算，避免普通检索窗口挤掉真正需要的片段。
- `memory_fragments`记录每条成功解析记忆的来源、哈希、需求ID及原文区间。Investigator保留记忆ID、来源ID与需求映射；完整原文只存一份，证据视图按来源展开一次。
- 片段数量和角色输入预算仍然有效；超过输入预算的来源标为omitted/deferred，保留原文及片段映射，不能算作已看见或已覆盖。

新增`memory_fragment_valid`（解析成功的记忆条数）、`memory_source_cache_hits`（同源片段复用已读原文次数）和`memory_candidate_sources`。在新流程中，`document_reads`按真实来源读取计数，`memory_source_valid`按唯一有效来源计数，来源回读覆盖率的分母也使用唯一候选来源，避免同源两片段导致覆盖率被误记成50%。

以上是need-first/2的片段保留修复；跨题入库去重与引用归因后续由need-first/3补充，不声称片段匹配是因果归因。

## 检查发现与对应修改

| 原路径的问题 | 新路径的处理 |
|---|---|
| 先按整题和选项召回，Planner处理候选预览 | Planner先读原题与全部选项，生成1至3个决定结论的知识需求，不接收历史记忆 |
| 生成需求后没有按需求再次召回 | 按每个需求的goal及关联条件生成检索向量，由记忆服务真正消费；不把完整题干、所有选项再拼进查询 |
| 策略、过程记录与证据一起争夺输入空间 | 本版本默认只召回证据记忆；过程保留在任务日志，避免把成功答题记录误当可复用知识 |
| 相似候选进入流程后没有明确影响检索决策 | Investigator提交逐需求的covered/partial/missing/conflicted/unusable及来源、条件判断和理由 |
| 每题仍固定进行教材检索 | covered且条件matched的需求跳过查询；仅对其余缺口执行原有工具计划 |
| Reasoner的逐需求审核没有接入实际答题分支 | 同一次答案响应提交need_reviews；程序检查需求覆盖、引句、真实可见性及适用性字段，再更新状态 |
| 补查绕过Investigator，难以体现调查角色 | Reasoner提出缺口及预期信息，返回Investigator规划，再回Reasoner核验；只允许查询指定缺口 |
| 整题主题索引容易匹配病例背景而非关系 | 新证据单位以需求goal索引，保留原题ID、原任务条件、来源哈希及原文引句 |
| 粗召回数量被叫作“实际展示” | memory_presented改为实际进入模型输入的记忆数；粗召回次数另记memory_coarse_candidate_occurrences |
| 未回读成功的候选与新检索来源同ID，被误算已投递 | 只有成功resolve并通过来源校验的记忆才能算delivered；新流程最终引用还要求Investigator接纳 |

## 实际执行顺序

```text
Planner：知识需求 + 关联条件
    ↓
程序：按需求生成embedding → 记忆服务检索 → 全局限额选源 → 回读校验
    ↓
Investigator：逐需求判定记忆适用性与覆盖
    ├─ 全部覆盖：不进行教材检索
    └─ 存在缺口：只检索缺口，支持条件备用查询
    ↓
Reasoner：答案 + 引句 + 逐需求核验 + 是否需要返回调查
    ├─ finish：完成；只晋升有来源支撑且条件匹配的需求证据
    └─ investigate：明确可能影响答案的缺口 → Investigator → 新证据 → Reasoner
```

正常仍是3次模型调用。一次返回调查需要额外2次调用；新配置最多3轮调查、7次模型调用。没有新可见证据时保留既有答案，不再调用一次Reasoner。资源不足、用量未知或输出截断时不启动无法完成核验的返回调查。程序不另调用“记忆评价Agent”。

Investigator的covered是暂定判断；Reasoner不认可时需求会重新打开，旧标记不能继续阻止检索。所有需求都必须提交覆盖与核验结果，错误引用和不完整审核不能产生“已完成”需求。审核更新整体校验后再提交，避免前半段成功、后半段失败造成部分修改。

## 共享状态和输入边界

题目、选项、原始来源、需求状态和工具观测保存在应用层工作空间；结构化模式经ACP-1传递任务对象引用和角色视图。Text走自然语言交接，使用同样的先规划、记忆覆盖和补查决策政策，不经过ACP。

成功回读后，记忆实际原文进入Investigator输入。其接纳的记忆才能默认进入Reasoner；不适用记忆若被排除，不能仅凭它与新检索来源ID相同就算复用。新检索结果可以独立进入Reasoner。返回调查时保留旧答案有效引句作为比较锚点，这不意味着重新认定该记忆适用。

只从记忆获得、尚未重新检索的来源，尽量只展示已存引句及前后各120字符上下文，保留原文偏移和has_more。程序只在该视图更短时采用，完整原文和校验哈希始终不变；新检索来源按普通证据视图处理。Reasoner可以根据具体缺口返回调查以获取额外上下文。这是有原文依据的局部展开，不是让云API凭state_id理解未见证据。

新证据记忆保留统一ID、来源Agent、创建时间、需求主题、摘要、原任务ID、原任务条件和真实来源引用。只晋升本次逐需求核验为supported、条件matched且引句实际可见的证据，不保存答案字母、评分标签或生成的医学结论。过程和策略仍在执行日志中，新流程暂不把它们晋升到默认召回库。来源校验和同一Reasoner判断不等于独立医学认证。

## 如何观察是否有效

| 指标 | 口径 |
|---|---|
| memory_covered_needs | 首轮调查暂定被记忆覆盖的需求数 |
| memory_supported_needs | 最后核验为supported、条件matched且实际引用已接纳记忆片段的需求数；不证明独占贡献 |
| memory_need_coverage_rate / memory_need_support_rate | 上述需求数除以知识需求总数 |
| memory_only_supported_tasks | 零教材检索、有实际记忆引用且所有需求获支持的任务数 |
| queries_skipped_covered | 被程序拦截的已覆盖查询数；不是全部节省检索数，因为模型也可以主动不提出查询 |
| memory_funnel | 召回、实际展示、选取、回读成功、投递、调查接纳、最终引用分别计数 |
| needs_review_errors | 需求审核不符合契约的次数；不能隐瞒为答题成功 |
| total_tokens、prompt_tokens、completion_tokens、model_calls | 使用真实API usage；向量字节、ACP字节单列，不能折算为聊天Token收益 |

`memory_presented`在旧流程表示Planner看见的候选，在新流程表示Investigator实际看见的记忆；新流程没有Planner候选接纳率和“展示后选择率”，相关旧字段为null，新增Investigator接纳率。每题每阶段按记忆ID去重，跨题累计。

先看同一workflow下structured_on相对structured_off的总Token净变化，同时查看答题率、未解决需求和协议错误。不能只看命中率，更不能把减少一次本地搜索直接等同节省一次模型调用。额外覆盖字段和二次调查可能增加Token，收益必须实测。

## 配置与验证

新配置：[compare-api-need-first.json](../configs/medical-related-30-v1/compare-api-need-first.json)。沿用固定30题、四条件和模型参数，使用新输出目录、独立记忆命名空间；条件内保持原题序。先运行本地验证：

```bash
bash scripts/run_medqa.sh test
bash scripts/run_medqa.sh preflight --config configs/medical-related-30-v1/compare-api-need-first.json --all
```

以下命令会调用已配置的付费API；本次没有执行：

```bash
bash scripts/run_medqa.sh compare --config configs/medical-related-30-v1/compare-api-need-first.json --all
```

离线测试包括真实ChatRoles输入边界、覆盖后零检索、只检索缺口、排除不适用记忆、核验失败不写记忆、返回调查可修改答案、预算限制、无新证据保留答案、10轮持久记忆和工具提交后崩溃续跑。

同源片段修复后已通过287项回归测试，其中34项为本流程专项。新增覆盖两需求四引句只读一次原文、Text与structured下游输入、来源读取额度用完后添加同源片段、原文变更只失败一次、缺失记忆引用不借用其他单元、预算不足显式延后、相同原文不同片段视图及提交后崩溃恢复。当前验证未调用付费生成API。

测试使用合成资料及固定输出，只证明实现和控制路径，不提供MedQA准确率或真实Token收益。固定30题已经用于开发，不应称作独立未见测试集，也不因共享记忆而自动成为存在硬前置依赖的任务链。
