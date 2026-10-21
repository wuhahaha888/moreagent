# 实验结果只读快照

保留完整与未完成批次。每个checkpoint使用SQLite在线备份采集；不同数据库分别采集，不保证跨库同一时刻，不能直接用作运行中恢复包。
原始目录继续保留本机，运行中实验会继续变化；GitHub展示的是采集时刻的固定快照。

| 批次 | 完成任务/目标 | 全题序齐全 |
|---|---:|---|
| [medical30-api-fourway-20261020-203209-v3](medical30-api-fourway-20261020-203209-v3/README.md) | 120/120 | True |
| [medical30-api-fourway-20261020-v2](medical30-api-fourway-20261020-v2/README.md) | 56/120 | False |
| [medical30-api-fourway-acp1-v1](medical30-api-fourway-acp1-v1/README.md) | 120/120 | True |
| [medical30-api-fourway-workspace-v1](medical30-api-fourway-workspace-v1/README.md) | 120/120 | True |
| [medical50-api-fourway-need-first-v2](medical50-api-fourway-need-first-v2/README.md) | 10/200 | False |
| [medical50-api-fourway-need-first-v3](medical50-api-fourway-need-first-v3/README.md) | 79/200 | False |

完整题序不等于所有题运行成功。失败题、未知usage和负结果均保留，不将部分实验与不同版本拼接成全量收益。
summary.csv汇总条件；各批次report.json、comparison.json、逐题jsonl.gz和可选数据库快照用于复核。
解压数据库：gzip -dk checkpoint.sqlite3.gz；解压轨迹：gzip -dk text_on.jsonl.gz。建议复制到临时目录审计。
