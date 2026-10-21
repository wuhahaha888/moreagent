# 50题四条件API实验

原30题不改，追加20道原创MedQA风格衍生题。各条件按原30→新增20运行，两个有记忆条件从独立空库开始在线积累。新题为开发期合成挑战集，未经过独立医学审题，不声称官方MedQA或独立盲测。

先规划→按需求召回→Investigator复核覆盖/仅查缺口→Reasoner审核/必要时返回调查，workflow=need_first/4。Text自然语言交接；structured采用ACP和role_views_v1，因此通信比较包含上下文组织差异。四条件相同模型、同题序、相同预算；max_output_tokens统一2600。绝不预灌连续性提示、答案或新增网页来源。

```bash
bash scripts/run_medqa.sh test
bash scripts/run_medqa.sh audit --config configs/medical-related-50-v2/compare-api-need-first.json --all
bash scripts/run_medqa.sh compare --config configs/medical-related-50-v2/compare-api-need-first.json --all
```

结果：experiments/results/medical50-api-fourway-need-first-v3。progress.md分开汇总原30与新增20，summary.md给完整机制指标，checkpoint.sqlite3保存响应和用量，两个记忆数据库各自持久化。监控仅本地读取，不调用API：

```bash
LD_LIBRARY_PATH=../miniconda3/envs/struckflow/lib ../miniconda3/envs/struckflow/bin/python -B scripts/monitor_medqa50.py experiments/results/medical50-api-fourway-need-first-v3
```

续跑须原源码和参数不变，用--resume --output原目录；已完成题包括失败题均不重跑。响应未落盘、usage未知会停止，避免自动重复计费。所有差异按实报告，不以是否答错筛题。
