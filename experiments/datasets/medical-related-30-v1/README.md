# 医学关联任务子集：三组各十题

按用户确认的 ID 提取原始 MedQA：贫血 → 肾脏 → 甲状腺，组内顺序固定。全局 30 轮，不打乱题干、选项或答案。

- `questions.md`：供人阅读的完整 30 道英文原题与选项，不含答案。
- `questions.jsonl`、`questions.csv`：不含答案的题目导出。
- `runner-benchmark.json`：现有加载器兼容文件，包含供评分器使用的标准答案；运行时 `Question.public()` 只暴露 ID、题干和选项。
- `order.json`：30 题固定顺序，指纹绑定新的 runner 文件。
- `groups/`：各组十题的无答案文件及顺序文件；每份顺序文件仍绑定根目录同一个 runner。
- `evaluation/answers.json`、`evaluation/tasks.jsonl`：仅供评分与数据管理。
- `evaluation/reuse-review.json`：继承的候选复用关系与待核对条件，仅供评估者使用，不得预灌到 Agent 或共享记忆；其中的候选关系不是实测复用记录。
- `groups.json`、`metadata.json`、`validation.json`、`checksums.json`：分组、来源、原文指纹与校验记录。
- `selection.json`：用户固定的题号、组内顺序及候选关联注释；重建无需旧 71 题目录。

这 30 题均已出现在此前 71 题实验中，是新整理的评测子集，不是全新未见的独立测试集。仅有主题和顺序不能证明有效复用，后续实验仍须记录历史来源、回读、实际使用、正确性及成本。

使用现有 moreagent 加载器时，将新实验配置的 `benchmark` 指向本目录 `runner-benchmark.json`，`ids` 指向本目录 `order.json`，并设 `split=all`。三组独立实验分别采用 `groups/<group_id>.order.json`，使用新的结果目录及记忆命名空间；需在方案中声明组间是否清空记忆。

数据包构建不调用模型，不修改运行时、实验结果或记忆库。当前四条件 API 配置见 `configs/medical-related-30-v1`。

复现：在 moreagent 根目录运行 `python -B experiments/datasets/medical-related-30-v1/build.py`。
