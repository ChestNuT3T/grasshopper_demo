# Grasshopper 技术顾问 — 幕墙参数化设计 AI 助手

面向幕墙参数化设计师的 RAG 智能问答系统，深度掌握 Rhino 8 原生电池及 Elefront/Human/Hare/Dawn 等插件知识，通过**意图驱动的分层检索策略**和**任务感知的提示词路由**，在 Grasshopper 电池选型、故障诊断、搭建指导等场景下提供精准技术问答。

## 产品文档

- [产品说明文档](product-doc.md) — 架构设计、LCEL 链编排、检索策略详解、提示词工程
- [产品需求文档](prd.md) — 用户画像、产品策略、功能规划、迭代路线图

## 技术栈

| 层级 | 技术 |
|------|------|
| 链编排 | LangChain 1.0.4 LCEL |
| LLM | DeepSeek v4-pro（OpenAI 兼容协议，支持 thinking 推理） |
| 向量数据库 | ChromaDB 1.3.5 |
| Embedding | DashScope text-embedding-v4（1024维） |
| Rerank | DashScope qwen3-vl-rerank |
| 后端 | FastAPI + Uvicorn（SSE 流式） |
| 前端 | React 18 + TypeScript + Vite + TailwindCSS |

## 项目结构

```
CW_AI/
├── .env.example           # 环境变量模板
├── .gitignore             # Git 忽略配置
├── requirements.txt       # Python 依赖
│
├── config/                # Python 配置加载模块
├── configs/
│   ├── app.yaml           # 应用配置
│   └── prompts.yaml       # 提示词模板
│
├── src/
│   ├── core/              # 核心基础设施（客户端、链、存储、日志）
│   └── modules/           # 业务模块（理解、检索、提示词）
│
├── web/
│   ├── backend/main.py    # FastAPI 后端
│   └── frontend/          # React 前端
│
├── data/
│   ├── RAG/               # 原始知识数据（PDF 手册，已含）
│   ├── chroma_db/         # 向量库（已含，无需重建）
│   └── processed/         # 预处理数据（已含，无需重建）
│
├── scripts/
│   └── build_index.py     # 知识库索引构建脚本（仅需更新数据时运行）
│
├── logs/                  # 日志（运行时生成）
└── chat_history/          # 会话历史存储（SQLite，运行时生成）
```

---

## 环境要求

- **Python 3.10+**
- **Node.js 18+**
- **DeepSeek API Key**（必填，LLM 推理）
- **DashScope API Key**（推荐，用于 Embedding 和 Rerank；未设置则 RAG 功能不可用）

---

## 快速开始

### 第 1 步：安装依赖

在项目根目录打开终端，依次执行：

**安装 Python 依赖：**

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**安装前端依赖：**

```bash
npm --prefix web/frontend install
```

### 第 2 步：配置环境变量

复制环境变量模板：

```bash
copy .env.example .env
```

编辑 `.env` 文件，填入 API Key：

```
DEEPSEEK_API_KEY=sk-你的DeepSeek密钥
DASHSCOPE_API_KEY=sk-你的DashScope密钥
```

其余配置项保持默认即可。

### 第 3 步：启动服务

需要**同时运行**后端和前端。打开两个终端，均在项目根目录下执行：

**终端 1 — 启动后端：**

```bash
python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**终端 2 — 启动前端：**

```bash
npm --prefix web/frontend run dev
```

### 第 4 步：访问应用

启动成功后，在浏览器打开：

- **前端页面**：http://localhost:3000
- **API 文档**：http://localhost:8000/docs

> **说明**：知识库（向量库 + 预处理数据 + PDF 手册）已包含在项目中，无需额外构建，启动后即可直接使用。
