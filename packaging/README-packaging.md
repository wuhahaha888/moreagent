# StruckFlow - openEuler 24.03-LTS-SP3 打包部署指南

## 项目概述

StruckFlow 是一个基于 LangChain + LangGraph 的结构化多智能体对话应用，包含：

- **Python 后端**：Flask API + 多 Agent 协作系统（LangChain/目录）
- **Vue 3 前端**：Element Plus UI（web/vue/）
- **Flask API 层**：封装后端供前端调用（web/flask/）

### 关键依赖

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| Python | >= 3.11 | openEuler 24.03 自带 Python 3.11 |
| Node.js | >= 18 | 前端构建 |
| PyTorch | >= 2.1 | Embedding 模型推理 |
| sentence-transformers | >= 3.0 | 文本向量化 |
| LangChain | >= 0.3 | LLM 框架 |
| Flask | >= 3.0 | API 服务 |
| Nginx | - | 反向代理 + 静态文件 |

---

## 方式一：快速部署（推荐）

使用一键部署脚本，适合开发和生产环境：

```bash
# 克隆项目
git clone https://github.com/Aloe-Becone/StruckFlow && cd StruckFlow

# 开发模式部署
sudo bash packaging/setup-openeuler.sh --dev

# 生产模式部署（含 systemd + nginx 配置）
sudo bash packaging/setup-openeuler.sh --prod

# 生产模式 + GPU 支持
sudo bash packaging/setup-openeuler.sh --gpu
```

部署脚本会自动：
1. 检测 openEuler 操作系统
2. 安装 Python3、Node.js、Nginx 等系统依赖
3. 创建 Python 虚拟环境并安装项目依赖
4. 构建 Vue 前端
5. 配置 systemd 服务和 Nginx 反向代理（生产模式）

---

## 方式二：RPM 包部署

### 1. 构建 RPM 包

在 openEuler 24.03-LTS-SP3 构建机上执行：

```bash
# 安装构建工具
sudo dnf install -y rpm-build rpmdevtools python3-devel nodejs npm rsync

# 构建 RPM
bash packaging/build-rpm.sh

# 跳过前端构建（如已构建）
bash packaging/build-rpm.sh --skip-frontend
```

构建产物位于 `packaging/struckflow-1.0.0-1.*.rpm`。

### 2. 安装 RPM 包

在目标机上执行：

```bash
sudo dnf install -y ./struckflow-1.0.0-1.noarch.rpm
```

安装后自动完成：
- 创建 `struckflow` 系统用户
- 创建数据目录 `/var/lib/struckflow/data/`
- 初始化 `.env` 配置文件
- 启用并启动 systemd 服务

### 3. 配置和启动

```bash
# 编辑 API 配置（必须填写 API Key）
struckflow-manage config

# 启动服务
struckflow-manage start

# 查看状态
struckflow-manage status

# 查看日志
struckflow-manage logs

# 运行诊断测试
struckflow-manage test
```

---

## 方式三：手动部署

### 1. 安装系统依赖

```bash
sudo dnf makecache --refresh
sudo dnf install -y python3 python3-pip python3-devel nginx git

# Node.js（前端构建需要）
sudo dnf install -y nodejs npm
```

### 2. 配置 Python 环境

```bash
cd /opt/struckflow
python3 -m venv .venv
source .venv/bin/activate
pip install -r LangChain/requirements.txt \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 构建前端

```bash
cd web/vue
npm install --registry=https://registry.npmmirror.com
npm run build
cd ../..
```

### 4. 配置环境变量

```bash
cp LangChain/.env.example LangChain/.env
vim LangChain/.env
```

必须配置的变量：
```env
OPENAI_API_KEY=你的API_KEY
OPENAI_BASE_URL=实际API地址
DEEPSEEK_API_KEY=你的API_KEY
DEEPSEEK_BASE_URL=实际API地址
EMBEDDING_MODEL_PATH=本地Embedding模型路径
```

### 5. 启动服务

```bash
# 开发模式
source .venv/bin/activate
python LangChain/main.py

# Flask API
python web/flask/app.py

# 前端开发服务器
cd web/vue && npm run dev
```

---

## GPU (CUDA) 支持

如果需要 GPU 加速 Embedding 模型推理：

```bash
# 1. 安装 NVIDIA 驱动和 CUDA Toolkit
# 参考：https://developer.nvidia.com/cuda-downloads

# 2. 安装 GPU 版 PyTorch
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. 验证
python -c 'import torch; print("CUDA available:", torch.cuda.is_available()); print("GPU count:", torch.cuda.device_count())'
```

---

## 目录结构（RPM 安装后）

```
/usr/lib/struckflow/          # 应用主目录
├── LangChain/                # Python 后端核心
│   ├── .env                  # 环境配置
│   ├── main.py               # CLI 入口
│   └── ...
├── web/flask/                # Flask API
│   └── app.py
└── venv/                     # Python 虚拟环境

/usr/share/struckflow/web/    # Vue 前端静态文件

/var/lib/struckflow/data/     # 数据目录
├── memories/                 # 记忆存储
└── chats/                    # 聊天记录

/etc/struckflow/              # 配置模板
/etc/nginx/conf.d/            # Nginx 配置
/usr/lib/systemd/system/      # systemd 服务
```

---

## 测试验证

### 功能测试

```bash
# 1. 环境诊断
struckflow-manage test

# 2. CUDA 测试
/usr/lib/struckflow/venv/bin/python -c 'import torch; print(torch.cuda.is_available())'

# 3. SearXNG 搜索测试
curl -G --data-urlencode 'q=example' --data-urlencode 'format=json' http://localhost:8082/search

# 4. API 健康检查
curl http://localhost:5000/api/health

# 5. 前端访问
curl http://localhost/
```

### 服务管理

```bash
struckflow-manage start      # 启动
struckflow-manage stop       # 停止
struckflow-manage restart    # 重启
struckflow-manage status     # 状态
struckflow-manage logs       # 日志
struckflow-manage config     # 配置
struckflow-manage pip <包名> # 安装 Python 包
```

---

## 常见问题

### Q: Node.js 安装失败
openEuler 24.03 默认仓库可能不含 Node.js 18+，可通过 NodeSource 安装：
```bash
curl -fsSL https://rpm.nodesource.com/setup_22.x | sudo bash -
sudo dnf install -y nodejs
```

### Q: PyTorch 安装缓慢
使用清华镜像：
```bash
pip install torch --index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu121/
```

### Q: Nginx 端口冲突
修改 `/etc/nginx/conf.d/nginx-struckflow.conf` 中的 `listen` 端口。

### Q: 服务启动失败
查看日志：`journalctl -u struckflow-backend -n 50`
常见原因：`.env` 配置缺失或 API Key 无效。
