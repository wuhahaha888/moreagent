# Moreagent / StruckFlow

面向多智能体协作的通信、非文本状态交换与共享记忆实验原型。Planner规划知识需求，Investigator检索缺口，Reasoner核验原文与条件并回答；三个角色使用同一模型服务和不同提示词。

当前任务流为 **原始MedQA 30题＋原创MedQA风格扩展20题，共50题**。四条件为自然语言/结构化通信×记忆关闭/开启，每条件50题，共200次任务。生成通过DeepSeek API，教材检索、重排、embedding与SQLite在本地执行。

## 数据与结果

- [完整50题](experiments/datasets/medical-related-50-v2/questions.md) · [新增20题](experiments/datasets/medical-related-50-v2/extensions.md) · [数据来源、连续性与局限](experiments/datasets/medical-related-50-v2/README.md)
- [实验结果索引](reports/README.md) · [2026-10-21固定快照](reports/experiment-snapshots/20261021-1023/README.md)：含逐题轨迹、账本和记忆库压缩快照；未完成批次明确标注，不拼接成全量成绩。
- [两批完整30题对照](reports/medical30-acp-workspace-comparison.md)：Workspace组合方案的structured无记忆总Token为134798，Text无记忆178089，观察下降24.31%；四条件均30/30正确。structured开启记忆为151409，较关闭记忆增加12.32%，未证明记忆净收益。
- [记忆审计](reports/medical30-memory-audit.md) · [50题实验前复核](reports/medical50-preflight-review.md)：缺陷、修正和验证边界均保留。

最新50题批次仍在执行，GitHub结果是标明采集时间的部分快照，不能视为200任务全部完成。完整历史批次、停止批次和负结果均保留；本机原始实验目录不删除、不覆盖。

原30题保持题干、选项、答案及顺序；新增为贫血7题、肾脏7题、甲状腺6题。新增题为本项目合成病例，非官方MedQA新增题，未经过独立医学审题或难度标定。连续性体现知识复用、条件变化和延迟回读，各题仍可独立求解；不声称同一患者病程或硬前置依赖。答案与人工复用提示不进入Agent输入或初始记忆。

## 系统实现

| 部分 | 实际职责 |
|---|---|
| 多Agent运行时 | Planner→Investigator→Reasoner顺序执行；Reasoner可提出具体缺口并返回Investigator。 |
| 通信与调度 | Text自然语言交接，structured采用ACP-1二进制位置参数、能力协商、对象引用与函数派发。模型工具/结果接口仍采用JSON。 |
| 共享工作状态 | 任务、选项、需求、证据和执行状态统一保存，structured按角色构造必要视图，减少重复字段。 |
| 非文本状态 | float32向量通过进程间Pipe交换，由记忆服务实际消费以检索相似片段；没有隐藏状态/KV注入。 |
| 共享记忆 | 先规划再按需求召回；同源只回读一次、不同片段分别保留；知识与任务来源分离、最终复用按原文区间归因。 |
| 评测与恢复 | 答案、证据、协议独立统计；真实API usage、通信、向量、工具与复用链；SQLite账本保存响应和待提交记忆。 |

检索与记忆服务是程序工具，不额外算作模型Agent。Textbooks使用BM25与MedCPT；来源匹配和模型自检不能代替独立医学判断。ACP-1是本项目协议，不是论文G²CP完整复现；structured收益还包含角色视图，不能只归因于数字编码。

## 安装与执行

要求Python 3.11+，依赖见[requirements.txt](LangChain/requirements.txt)。还需预先准备教材、BM25索引、MedCPT、Qwen3-Embedding及相应Java/检索环境，见[环境准备](docs/SETUP.md)。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r LangChain/requirements.txt
cp LangChain/.env.example LangChain/.env
export STRUCKFLOW_PYTHON="$PWD/.venv/bin/python"
```

在LangChain/.env填写DEEPSEEK_API_KEY，密钥不提交。当前配置为deepseek-flash，实际响应模型、参数和用量记录在manifest及账本中。

```bash
# 不调用生成API的检查
bash scripts/run_medqa.sh test
bash scripts/test_memory.sh
bash scripts/run_medqa.sh audit \
  --config configs/medical-related-50-v2/compare-api-need-first.json --all

# 完整四条件，会调用API；新批次必须使用尚不存在的结果目录
bash scripts/run_medqa.sh compare \
  --config configs/medical-related-50-v2/compare-api-need-first.json \
  --all --output experiments/results/medical50-new-run

# 只读进度与自动汇总，不另调API
python -B scripts/monitor_medqa50.py experiments/results/medical50-new-run

# 中断恢复须使用原源码、参数和数据库；未知usage会阻止普通续跑
bash scripts/run_medqa.sh compare --resume \
  --output experiments/results/medical50-new-run
```

关闭终端时需要nohup或tmux保持进程；API断连仍可能触发保护停止。不要为续跑删除原记忆库、篡改未知用量或将失败题强行标成成功。

## 仓库结构

```text
LangChain/agents/             三角色与提示词
LangChain/medqa/              协议、状态、检索、证据、评测与检查点
LangChain/memory/             embedding与持久化基础组件
LangChain/test/               核心离线回归
configs/medical-related-50-v2/ 当前50题API配置
configs/medical-related-30-v1/ 历史及对照配置
experiments/datasets/         原30题与扩展50题、来源及构建脚本
reports/                     实验汇总、负结果与可下载的固定快照
scripts/                     运行、汇总、审计、快照与恢复工具
web/、packaging/              原通用界面与部署工具
```

原通用Web尚未接入当前50题实验。模型权重、密钥、虚拟环境和实时数据库不纳入Git；公开的压缩结果快照位于reports，不依赖运行中目录。已清理被正式实现替代的示例与手动探针，保留必要回归测试和全部本机数据/结果。

[系统设计](docs/ARCHITECTURE.md) · [通信实现](docs/COMMUNICATION_IMPLEMENTATION.md) · [共享状态](docs/SHARED_WORKSPACE.md) · [需求驱动记忆](docs/NEED_FIRST_MEMORY.md) · [实验设计](docs/EXPERIMENTS.md) · [仓库清理](docs/REPOSITORY.md)
