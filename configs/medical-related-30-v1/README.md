# 三组 30 题 API 四条件实验

自然语言/ACP-1通信对照使用compare-api-acp1.json，已完成medical30-api-fourway-acp1-v1批次；共享角色视图批次也已完成，见[结果对照](../../reports/medical30-acp-workspace-comparison.md)。当前主线为[50题配置](../medical-related-50-v2/README.md)。下面v3是修正前的格式对照历史，不可用当前源码普通续跑或当成ACP-1结果。

数据：用户确认的贫血、肾脏、甲状腺三组，各 10 题。组间及组内均保持指定顺序。

当前批次使用 `compare-api-v3.json`，输出为 `experiments/results/medical30-api-fourway-20261020-203209-v3`，于北京时间 2026-10-20 20:32在宿主环境启动。此批次使用已修复的 JSON 解析器，从独立空库重新执行完整四条件，不导入旧答案或旧记忆。启动信息、只读进度和源码快照位于同名 `experiments/validation/` 目录；最新批次指针为 `experiments/validation/medical30-api-current.json`。

v1 在受限环境首个请求发生 APIConnectionError 后停止；v2 在宿主环境完成 56/120 次任务记录，随后因一条 API 响应缺失停止。原始结果和未知用量记录分别保留，不并入 v3。v2 使用修复前的源码，不能直接以当前版本普通 `--resume` 续跑，详见 `docs/JSON_RECOVERY.md`。

四条件为 `text_off`、`structured_off`、`text_on`、`structured_on`，每个条件执行同样的 30 题，共 120 次任务执行。使用现有顺序 compare 运行器，条件内串行，条件间也串行；实际条件顺序按 condition_seed=42 固定并写入结果。

生成模型固定为官方 DeepSeek API 的 deepseek-flash，读取既有 DEEPSEEK_API_KEY；配置中不保存密钥，不使用本地 Qwen。教材、BM25/MedCPT 检索、本地 embedding、三角色提示、needs 记忆策略与调用预算沿用现有实现，不在本次修改。

每个条件使用独立数据库，不读取之前 71 题实验或其他条件的记忆；开启记忆的条件从空库开始，沿 30 题序列在线积累，组间不清空。因此这是全局连续 30 轮设计，不是三个各自冷启动的实验。组级结果可按数据包 groups.json 分解。

数据已参加过旧 71 题实验，是新整理的评测子集，不能称为全新未见测试集。候选复用注释仅供评估，不作为 Agent 输入或预置记忆。

从 moreagent 根目录运行：

```bash
# 静态校验，不调用生成 API。
bash scripts/run_medqa.sh audit --config configs/medical-related-30-v1/compare-api-v3.json --all
# 会调用已配置 API，需使用未存在的输出目录。
bash scripts/run_medqa.sh compare --config configs/medical-related-30-v1/compare-api-v3.json --all
# v3 已启动；不要重复启动。仅在中断且代码与配置指纹一致时恢复原目录。
bash scripts/run_medqa.sh compare --resume \
  --output experiments/results/medical30-api-fourway-20261020-203209-v3
```

每题最多 4 次模型调用，输出上限 1800 token；每条件总保护预算 60 万 token，全实验 240 万 token。预算在操作边界检查，不是严格账单上限；以真实 API usage 报告实际成本。未知响应不自动重发。保留失败题、协议问题和负收益，不覆盖旧结果。
# 共享工作状态实验

`compare-api-workspace.json` 使用同一固定30题和四条件，structured额外启用`role_views_v1`。结果独立写入`experiments/results/medical30-api-fourway-workspace-v1`，记忆命名空间独立。Text输入与原基线相同；structured收益属于协议及角色上下文的组合方案。实现边界见`docs/SHARED_WORKSPACE.md`。

## 先规划、按需求复用记忆

`compare-api-need-first.json`设置`workflow=need_first`：Planner先生成需求，之后按需求召回证据，Investigator只查未覆盖的缺口，Reasoner逐需求核验并可返回调查。两个通信模式使用相同流程。正常3次模型调用，最多3轮调查/7次调用；其余单题及整批Token预算保持配置值。新目录为`experiments/results/medical30-api-fourway-need-first-v1`，记忆命名空间独立。没有启动该配置的付费实验。完整说明见[需求驱动记忆](../../docs/NEED_FIRST_MEMORY.md)。
