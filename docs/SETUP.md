# 环境与数据准备

当前主线使用API生成，仍需要本地教材检索和embedding。源码和30题数据包可以提交GitHub；权重、教材索引、API密钥、虚拟环境及运行数据库保留在仓库外或忽略目录中。

## 环境

- 应用Python：3.11+，安装 `LangChain/requirements.txt`。
- 检索Python：能运行BM25/Pyserini工作进程的独立环境，当前部署使用已有medrgag环境。
- Java：与现有BM25/Pyserini索引兼容的运行时，当前部署使用Java21。
- MedCPT Cross Encoder与Qwen3-Embedding本地权重；CLI离线读取，不自动下载。
- 模型API密钥填在 `LangChain/.env`，以 `.env.example` 为模板。

`scripts/run_medqa.sh`默认查找当前机器的struckflow环境。其他机器需先设置 `STRUCKFLOW_PYTHON` 为已安装依赖的Python绝对路径，不必照搬本机Conda目录。

## 路径

| CLI参数 | 目录或文件要求 |
| --- | --- |
| `--benchmark` | 已附的 `experiments/datasets/medical-related-30-v1/runner-benchmark.json`。 |
| `--ids` | 同数据包的 `order.json`，指纹必须与benchmark一致。 |
| `--textbooks` | `chunk/*.jsonl`及`index/bm25/`，必须事先准备。 |
| `--retrieval-python` | BM25检索环境Python绝对路径。 |
| `--java-home` | 含`bin/java`的Java根目录。 |
| `--ranker` | MedCPT Cross Encoder完整本地模型目录。 |
| `--embedding` | Qwen3-Embedding完整本地模型目录。 |

默认数据布局是项目的相邻目录 `../medrgag-data/` 和 `../miniconda3/envs/medrgag/`，embedding位于 `models/Qwen3-Embedding-0.6B`。路径不同时，通过JSON配置或上述CLI参数覆盖；CLI显式参数优先。教材与索引来源沿用本地MedRGAG准备流程，本仓库不包含下载/建索引的完整流水线。

API实验配置显式指定模型与端点，密钥只从 `DEEPSEEK_API_KEY` 读取。运行前先执行 `audit --config ... --all --output 一个新目录`；静态audit不发生成请求，也不保证API实时可达。

## 测试

```bash
bash scripts/run_medqa.sh test
bash scripts/test_memory.sh
```

上述回归使用替身模型，验证角色契约、协议、真实SQLite/IPC、证据、预算和恢复等机制，不代表真实医学准确率或性能收益。旧CUDA/SearXNG手动探针已清理；必要的部署诊断命令见packaging文档。
