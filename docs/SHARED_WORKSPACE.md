# 顺序协作共享工作状态

实现版本：`moreagent.workspace/1`；配置开关：`workspace_policy=role_views_v1`。选择应用层内存对象及原有SQLite检查点，不新增POSIX共享内存、共享显存、KV注入或模型调用。适用于当前Planner → Investigator → tools → Reasoner的顺序执行。

## 为什么选择这个方案

当前目标是减少实际模型输入和输出Token。替换Pipe为共享内存不会自行减少远程API输入，改造模型内部状态则需要控制本地推理引擎。复用现有工具、原文证据处理和恢复机制，改造角色输入，工程风险较小，能够用当前固定MedQA题目直接测量。

## 实际行为

- `Workspace`持有规范任务状态和证据表；原题、选项保持不可变。Planner提交校验后的规划字段，Investigator的执行计划保存为状态，Reasoner提交最终结果。变更记录包含角色、版本、字段和内容哈希；失败结束由Runtime标识，不假称是模型产物。
- 工具仍直接消费查询、需求ID、前驱ID和执行条件；检索结果不增加模型总结调用。
- ACP的完整任务对象按内容去重。模型的任务视图通过独立引用选择，接收端不会将原题强制补回缩略视图。完整任务与局部投影分别计入对象登记字节。
- `RoleViewBuilder`只在structured条件启用。Text条件的提示和输入保持原自然语言交接，由捕获实际模型请求的测试验证。

| 角色 | 读取 | 不重复投递 |
|---|---|---|
| Planner | 原题、全部选项、记忆候选预览、有效反馈 | 执行预算和空字段 |
| Investigator | 原题或合格原文条件缩略、全部选项、需求ID与目标、必要记忆与原文、检索预算 | 重复条件、需求关联映射、执行状态表、无用空字段 |
| Reasoner | 完整原题与全部选项、相同的证据视图、需求ID与目标、已查查询、缺口、冲突、补查反馈与必要预算 | 重复规划条件、任务支持映射、历史记忆记录、工具日志与执行控制字段 |

Investigator只在条件均为原题逐字片段、数字齐全、显式否定句保留且估计文本代价更低时采用缩略。该检查是词面检查，不证明语义完整；否则回退完整题干。全部选项始终保留。不会为了凑节省比例强制删题。Reasoner始终对照完整原题作答。

未改变教材证据正文的选取算法、检索工具参数、查询执行条件、引用核验、记忆准入和聊天调用上限。向量仍由已有ACP句柄和二进制记忆通道实际消费，不在模型提示中展开浮点数组。

## 恢复与统计

每步检查点保存工作空间版本、变更历史和任务哈希；原规范状态仍保存于现有控制快照。恢复沿用模型响应与工具结果重放，不会额外付费总结。源码、配置和模型提示变化仍阻止混用原检查点。

新增指标：`workspace_views`、`workspace_input_bytes_before`、`workspace_input_bytes_after`、`workspace_brief_views`、`workspace_patch_count`。这些字节是同次模型调用投影前后的正文大小，不能代替API Token。实际输入、输出和用量继续保存在各角色事件中。

四条件实验将structured定义为“ACP＋共享状态角色视图”的组合方案；不能把全部收益归因于二进制编码。Text与structured各自比较开启/关闭跨任务记忆。旧ACP实验单独保存，不合并分数。

## 运行

```bash
bash scripts/run_medqa.sh test
bash scripts/run_medqa.sh audit --config configs/medical-related-30-v1/compare-api-workspace.json --all
bash scripts/run_medqa.sh compare --config configs/medical-related-30-v1/compare-api-workspace.json --all
```

固定三个主题各10题、四条件共120次任务；每条件独立数据库，均从空记忆开始。顺序保持anemia → renal → thyroid，组间不重置该条件的记忆。相同题目属于开发中反复使用的固定评测流，不称为未见测试集。
