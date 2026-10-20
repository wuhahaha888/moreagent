# Moreagent / StruckFlow

面向多智能体协作的通信、非文本状态交换与共享记忆实验原型。当前主线以三类模型角色协作处理医学选择题，检索真实教材证据，并在连续任务中积累、核验和复用历史记忆。

当前评测数据固定为 **MedQA 三组各 10 题，共 30 题**：贫血、肾脏、甲状腺。比较纯文本/结构化输入 × 记忆关闭/开启四条件，共 120 次任务执行。生成模型通过 DeepSeek API 调用；检索、重排、embedding 和记忆存储在本地执行。

## 系统做什么

| 部分 | 实现 |
| --- | --- |
| 多 Agent 运行时 | Planner 规划需求；Investigator 制定查询和工具计划；Verifier/Reasoner 核验原文并回答。三个角色使用同一模型服务和不同提示词。 |
| 通信与调度 | G2CP 的 Control / Semantic / Precise 承载、能力描述、路由与预算；相同角色载荷分别编码为可读文本或紧凑 JSON。 |
| 非文本状态 | 运行时与独立记忆服务通过 Pipe 传输 float32 向量字节，接收端恢复后用于余弦检索。 |
| 共享记忆 | SQLite 保存证据、检索经验和任务过程，含来源角色、时间、主题、摘要和来源版本；后续任务按 ID 回读。 |
| 评测与恢复 | 分别统计答案、证据、协议结果、真实 API usage、通信量、向量负载、工具次数、耗时和复用链；SQLite 检查点支持中断恢复。 |

检索器和记忆服务是程序工具，不计入三个模型 Agent。工具使用 Textbooks 的 BM25 索引与 MedCPT 重排，不把模型生成的补充知识当作教材证据。

## 当前任务与实验

| 顺序 | 任务组 | 题数 |
| --- | --- | ---: |
| 1 | 贫血分类、检验解释与条件变化 | 10 |
| 2 | 蛋白尿、血尿来源与肾脏机制鉴别 | 10 |
| 3 | 甲状腺功能、母婴条件与用药边界 | 10 |

[完整题目](experiments/datasets/medical-related-30-v1/questions.md) · [数据说明](experiments/datasets/medical-related-30-v1/README.md) · [API 实验配置](configs/medical-related-30-v1/README.md)

这是知识关联的连续任务流：组内保留顺序，开启记忆时跨题积累，并允许后题复用前题成果。各题仍可独立求解，不声称是同一患者的连续病程或严格前置依赖工作流。30 题均曾出现在旧 71 题实验中，是固定评测子集，不是全新未见测试集。标准答案与候选复用注释不进入 Agent 输入或初始记忆。

四条件使用相同题序、模型、提示词、工具与预算。各条件使用独立数据库；记忆开启的条件从空库开始，沿 30 题在线积累，组间不清空。详见 [实验设计与指标](docs/EXPERIMENTS.md)。

当前 30 题 API 实验结果写入本机 `experiments/results/medical30-api-fourway-20261020-v2/`，该目录不提交到 Git。全量结束前不发布最终收益结论。旧 71 题结果作为[历史参考](reports/medical71-reference/README.md)保留，其中记忆增加了成本，不能宣称已经证明普遍加速。

## 安装与运行

要求 Python 3.11+、可用的兼容模型 API，以及预先准备的教材、BM25 索引、MedCPT 与 embedding 权重。Python 依赖见 [requirements.txt](LangChain/requirements.txt)。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r LangChain/requirements.txt
cp LangChain/.env.example LangChain/.env
export STRUCKFLOW_PYTHON="$PWD/.venv/bin/python"
```

在 `LangChain/.env` 填写实际 API 配置。当前实验配置显式使用 `deepseek-flash`、`https://api.deepseek.com` 与 `DEEPSEEK_API_KEY`。模型名、端点和请求上限以实际配置和响应记录为准。密钥不提交到 Git。

教材检索使用独立 Python/Java 环境及已构建索引；仅安装上述 Python 包不会自动下载教材或构建 BM25。目录要求及可覆盖参数见 [环境准备](docs/SETUP.md)。

```bash
# 静态检查，不调用生成 API；新实验选择尚不存在的输出目录。
bash scripts/run_medqa.sh audit \
  --config configs/medical-related-30-v1/compare-api.json --all \
  --output experiments/results/medical30-api-new

# 四条件全量，会调用 API。
bash scripts/run_medqa.sh compare \
  --config configs/medical-related-30-v1/compare-api.json --all \
  --output experiments/results/medical30-api-new

# 中断后沿原参数、题序和数据库恢复。
bash scripts/run_medqa.sh compare --resume \
  --output experiments/results/medical30-api-new

# 离线回归，使用替身模型，不产生生成 API 费用。
bash scripts/run_medqa.sh test
bash scripts/test_memory.sh
```

数据包已经包含 30 题与加载器文件，不需要下载整个原题库才能执行这套子集。重建数据包时才需要本地原始 `MedRGAG/benchmark.json`；脚本校验来源指纹并保留原文、选项及标准答案。

## 仓库结构

```text
LangChain/agents/             模型角色与提示词
LangChain/medqa/              当前运行时、检索、证据、记忆、评测与检查点
LangChain/protocol/           公共协议承载
LangChain/memory/             embedding 和存储基础组件
LangChain/test/               离线单元及集成测试
configs/medical-related-30-v1/ 当前 API 四条件配置
experiments/datasets/         固定 30 题及重建脚本
reports/                     可提交的精简历史报告
scripts/                     运行、审计、汇总和可选本地模型工具
web/                         原通用聊天界面
packaging/                   原通用 Web 的部署工具
```

`web/` 仍连接原通用工作流，尚未接入当前 30 题三 Agent 实验。`packaging/` 的 Web 部署记录也不等于当前评测路径已经完成目标操作系统验收。

[系统设计](docs/ARCHITECTURE.md) · [实验设计](docs/EXPERIMENTS.md) · [中断恢复](docs/medqa-resume.md) · [清理与提交范围](docs/REPOSITORY.md)

## 结果边界

文本与结构化对比是相同角色输入的表示对照，两者输出均采用 JSON 决策契约。向量 IPC 没有向聊天模型注入隐藏状态，也不自动减少模型读取证据的 Token。来源哈希和引句匹配不能代替语义正确性；记忆命中、读取、实际引用、答对与净收益分别报告。
