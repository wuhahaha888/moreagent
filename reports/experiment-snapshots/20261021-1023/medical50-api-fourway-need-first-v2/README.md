# medical50-api-fourway-need-first-v2

部分结果快照，不能作为全量收益结论。采集时间：2026-10-21T10:22:57.998558+08:00

| 条件 | 完成/目标 | 答对 | 总Token | 运行失败 | usage完整 |
|---|---:|---:|---:|---:|---|
| text_off | 0/50 | 0 | 0 | 0 | False |
| structured_off | 0/50 | 0 | 0 | 0 | False |
| text_on | 10/50 | 6 | 51633 | 4 | True |
| structured_on | 0/50 | 0 | 0 | 0 | False |

snapshot.json记录实际账本状态；运行中pending请求不能视为失败或零费用。
report按完成题统计，账本还可能有未完成题的已保存调用；未知usage保留为未知。
jsonl.gz含逐题完整轨迹；sqlite3.gz为各库独立只读备份，可解压审计，不作为直接续跑入口。
原始实验目录、数据库、日志均保留本机，导出不停止运行，不改写结果。
