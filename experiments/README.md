# 数据集与本机实验

- datasets/medical-related-30-v1：原MedQA三组各10题，完整保留。
- datasets/medical-related-50-v2：原30题不变，追加20道原创MedQA风格题，含评估侧答案、来源与候选连续性注释。
- results/：本机原始实验、检查点、记忆数据库和轨迹，保持原位并由Git忽略；运行期间不移动、不删除。
- runlogs/：本机日志与PID记录，不提交Git。

GitHub可下载的固定结果在[reports/experiment-snapshots](../reports/experiment-snapshots/20261021-1023/README.md)。SQLite在线备份保证每个库单独一致，不保证多数据库同一时刻，不能当成直接续跑包。

只读导出新快照，不调用API、不改动原始运行：

```bash
bash scripts/run_medqa.sh test
LD_LIBRARY_PATH=../miniconda3/envs/struckflow/lib \
  ../miniconda3/envs/struckflow/bin/python -B scripts/export_medqa_results.py \
  --output reports/experiment-snapshots/一个新目录 --include-databases
```

当前50题配置见configs/medical-related-50-v2。历史部分结果、未知用量与负结果均保留，不按成绩选删实验。
