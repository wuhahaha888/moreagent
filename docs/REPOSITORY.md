# 仓库清理与GitHub提交范围

## 2026-10-21更新

当前主线为原30题加原创20题的50题四条件实验。此次整理保留所有数据集、原始实验结果与当前运行进程，未移动或删除experiments/results。

清理7个已被正式实现或常规诊断替代的脚本：

- LangChain/test/test-cuda.py、test-searx.py：旧硬件/搜索手动探针，部署诊断改用直接命令。
- scripts/numeric_protocol_demo.py、demo_medqa_handoffs.py：独立协议演示和重复fixture smoke，已有正式ACP/交接集成回归。
- scripts/probe_qwen_local.py、benchmark_qwen_local.py：旧本地模型调试与合成输出测速，当前API主线未使用。
- scripts/audit_medqa_numeric_tokens.py：探索期字段编号消融，不代表当前通信路径的实际收益。

这些文件的副本和SHA256已保存在本机仓库外：`/data/wanghc/moreagent-cleanup-archive-20261021-102238`，不随Git推送。保留核心协议、工作状态、共享记忆、数据集和恢复回归，以及当前运行/汇总/恢复工具。

## 数据与结果

- 两套数据集共38个文件，清理前后SHA256完全一致。
- 原始实验目录保留本机，Git忽略results、runlogs、密钥、模型和环境。
- GitHub新增[6批次固定快照](../reports/experiment-snapshots/20261021-1023/README.md)，包含完整与未完成批次、逐题压缩轨迹、manifest和SQLite审计备份。65项工件校验通过，各数据库完整性检查通过。
- 当前50题仍在运行；采集时79/200，明确作为部分结果。不同版本、不同批次不拼接成全量收益。
- 数据库逐个在线备份，不中断原运行，不保证多个库同一时刻，不能直接作为跨库续跑包。

## 验证与提交

294项MedQA回归、16项通用记忆回归、6项恢复回归通过，共316项；清理、导出、验证没有额外生成API请求。检查提交文件和压缩结果中是否含密钥及超大文件。实时日志/PID不纳入Git，实验快照放在reports显式版本管理。

远端使用private指向的wuhahaha888/moreagent；origin仍保留原上游地址。不改写远程历史、不强制推送、不覆盖队友提交。

验证记录见[cleanup-validation-20261021.json](../reports/cleanup-validation-20261021.json)。

## 历史归档

2026-10-20清理的历史文件保存在`/data/wanghc/moreagent-cleanup-archive-20261020-172500`，原验证见reports/cleanup-validation-20261020.json。该历史归档继续保留；本次不重复清理数据或实验结果。
