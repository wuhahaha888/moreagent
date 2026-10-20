# 三组 30 题 API 四条件实验

数据：用户确认的贫血、肾脏、甲状腺三组，各 10 题。组间及组内均保持指定顺序。

当前输出为 `experiments/results/medical30-api-fourway-20261020-v2`。v1 在受限环境首个请求发生 APIConnectionError 后自动停止，未获得生成响应；原始结果与未知用量记录单独保留，不并入 v2。v2 在宿主环境执行。

四条件为 `text_off`、`structured_off`、`text_on`、`structured_on`，每个条件执行同样的 30 题，共 120 次任务执行。使用现有顺序 compare 运行器，条件内串行，条件间也串行；实际条件顺序按 condition_seed=42 固定并写入结果。

生成模型固定为官方 DeepSeek API 的 deepseek-flash，读取既有 DEEPSEEK_API_KEY；配置中不保存密钥，不使用本地 Qwen。教材、BM25/MedCPT 检索、本地 embedding、三角色提示、needs 记忆策略与调用预算沿用现有实现，不在本次修改。

每个条件使用独立数据库，不读取之前 71 题实验或其他条件的记忆；开启记忆的条件从空库开始，沿 30 题序列在线积累，组间不清空。因此这是全局连续 30 轮设计，不是三个各自冷启动的实验。组级结果可按数据包 groups.json 分解。

数据已参加过旧 71 题实验，是新整理的评测子集，不能称为全新未见测试集。候选复用注释仅供评估，不作为 Agent 输入或预置记忆。

从 moreagent 根目录运行：

```bash
# 静态校验，不调用生成 API。
bash scripts/run_medqa.sh audit --config configs/medical-related-30-v1/compare-api.json --all
# 会调用已配置 API，需使用未存在的输出目录。
bash scripts/run_medqa.sh compare --config configs/medical-related-30-v1/compare-api.json --all
# 仅在既有执行中断后使用，不另建记忆库。
bash scripts/run_medqa.sh compare --resume \
  --output experiments/results/medical30-api-fourway-20261020-v2
```

每题最多 4 次模型调用，输出上限 1800 token；每条件总保护预算 60 万 token，全实验 240 万 token。预算在操作边界检查，不是严格账单上限；以真实 API usage 报告实际成本。未知响应不自动重发。保留失败题、协议问题和负收益，不覆盖旧结果。
