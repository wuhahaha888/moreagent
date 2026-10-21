# 实验报告索引

- [最新50题结果与问题（部分快照）](medical50-latest.md)：采集时79/200，保留协议失败与未回答，不报告全量收益。
- [完整结果快照索引](experiment-snapshots/20261021-1023/README.md)：6个批次，含逐题jsonl.gz、SQLite压缩备份、manifest、条件汇总和SHA256。原始实验数据仍保留本机。
- [30题ACP与Workspace两批完整对照](medical30-acp-workspace-comparison.md)：各120任务；通信组合方案成本下降，记忆净收益未证实。
- [30题记忆审计](medical30-memory-audit.md)：实际召回、投递、引用与额外开销。
- [记忆系统复核与缺陷](memory-system-review-20261021.md) · [50题实验前检查](medical50-preflight-review.md)。
- [旧71题参考](medical71-reference/README.md)：保留历史成本增加、解析失败与结果边界。

每批结果与配置、题序、来源指纹相对应。快照中的完整题序包含失败题，不等于全部任务成功。未完成批次不拼接为完整实验，未知usage不按零成本处理。快照只用于审计，不是运行中多数据库的同步恢复包。
