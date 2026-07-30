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

---

### 1. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```bash
npm --prefix web\frontend install
```

### 2. 配置环境变量

编辑 `.env`，填入：
```
DEEPSEEK_API_KEY=sk-xxx
DASHSCOPE_API_KEY=sk-xxx
```

```bash
copy .env.example .env
```

### 3. 构建知识库索引（首次运行需要）

```bash
python scripts/build_index.py
```

### 4. 启动服务

**终端 1 — 后端：**
```bash
python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**终端 2 — 前端：**
```bash
npm --prefix web\frontend run dev
```

**访问地址：**
- 前端页面：http://localhost:3000
- API 文档：http://localhost:8000/docs
