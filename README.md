# BZ-RAG 知识库问答系统

基于 RAG（检索增强生成）架构的知识库问答系统，从帮助中心自动抓取文档，支持纯向量检索和混合检索两种方式，集成查询优化、Reranker 重排序、多轮对话等高级特性，结合大模型生成回答。

## 架构概览

```
                          ┌─────────────────────────────────────────┐
                          │            知识库构建（ETL）              │
                          │  网页爬取 → 图片OCR → 文本切分 → 向量化入库  │
                          └─────────────────────────────────────────┘
                                            │
                                            ▼
┌──────────────┐    ┌──────────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐
│   用户提问    │ → │   查询优化    │ → │  混合检索  │ → │ Reranker │ → │ LLM 回答  │
└──────────────┘    └──────────────┘    └──────────┘    └─────────┘    └──────────┘
```

系统提供三套向量数据库实现（Chroma / Milvus / Qdrant），每套各有不同的查询优化策略：

| 向量数据库 | 部署方式 | 混合检索 | 查询优化 | Reranker | 多轮对话 |
|-----------|---------|---------|---------|----------|---------|
| Chroma | 本地嵌入 | 手动 BM25 + Python 端 RRF | 多查询扩展 | BGE-Reranker（本地） | - |
| Milvus | Docker Compose | 内置 BM25 + 服务端 RRF | 历史感知查询改写 | 智谱 Rerank（API） | 滑动窗口 |
| Qdrant | Docker 单容器 | 手动 BM25 + 服务端 RRF | 关键词补充 | - | - |

## 目录结构

```
BZ-RAG/
├── app/
│   ├── chroma/                        # Chroma 向量数据库实现
│   │   ├── knowledge_build.py         # 知识库构建（ETL + OCR）
│   │   ├── vector_search.py           # 纯向量检索 + RAG
│   │   ├── vector_search_agent.py     # 纯向量检索 Agent 模式
│   │   ├── hybrid_search.py           # 混合检索（多查询扩展 + RRF + Rerank）
│   │   ├── hybrid_search_agent.py     # 混合检索 Agent 模式
│   │   ├── bm25_index.py             # BM25 索引构建与缓存
│   │   ├── rrf.py                    # RRF 融合排序算法
│   │   └── db/                       # Chroma 本地持久化数据
│   ├── milvus/                        # Milvus 向量数据库实现
│   │   ├── knowledge_build.py         # 知识库构建（ETL + OCR + BM25 Function）
│   │   ├── vector_search.py           # 纯向量检索 + RAG
│   │   ├── hybrid_search.py           # 混合检索（历史感知改写 + RRF + Rerank + 多轮对话）
│   │   └── docker-compose.yml         # 服务编排（etcd + MinIO + Milvus + Attu）
│   └── qdrant/                        # Qdrant 向量数据库实现
│       ├── knowledge_build.py         # 知识库构建（ETL + OCR + BM25 稀疏向量）
│       ├── vector_search.py           # 纯向量检索 + RAG（langchain_qdrant）
│       ├── hybrid_search.py           # 混合检索（关键词补充 + Prefetch RRF）
│       ├── bm25.py                   # BM25 稀疏向量（构建/保存/查询）
│       └── bm25_meta.json            # BM25 词汇表/IDF 持久化（自动生成）
├── common/                            # 公共模块
│   ├── image_ocr.py                   # GLM-4V-Flash 图片文字识别
│   ├── ocr.py                        # 智谱 OCR 工具
│   ├── reranker.py                   # BGE-Reranker 本地模型重排序
│   ├── zhipu_rerank.py               # 智谱 Rerank API 重排序
│   ├── query_rewriter.py             # LLM 查询改写
│   ├── query_expansion.py            # 多查询扩展
│   ├── keyword_expansion.py          # 同义词关键词补充
│   └── contextual_rewriter.py        # 历史感知查询改写（多轮对话）
├── .env                              # 环境变量配置
├── .env.example                      # 环境变量模板
├── requirements.txt                  # Python 依赖
└── README.md
```

## 环境依赖

- Python 3.10+（推荐 3.12，3.14 部分依赖兼容性不佳）
- Docker（Milvus / Qdrant 方案需要）

### 外部服务

| 服务 | 用途 | 配置项 |
|------|------|--------|
| MiniMax API | LLM 对话生成 | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `MODEL_ID` |
| 智谱 AI | 文本 Embedding（embedding-3） | `ZHIPUAI_API_KEY` |
| 智谱 AI | 图片 OCR（GLM-4V-Flash，免费） | `ZHIPUAI_API_KEY` |
| 智谱 AI | Rerank 重排序（Milvus 方案） | `ZHIPUAI_API_KEY` |

### 本地模型

| 模型 | 大小 | 用途 | 自动下载 |
|------|------|------|---------|
| BAAI/bge-reranker-base | 1.1GB | Rerank 重排序（Chroma 方案） | 首次运行时从 HF 镜像下载 |

## 配置说明

复制 `.env.example` 为 `.env`，填入对应的 API Key：

```bash
cp .env.example .env
```

```env
# LLM（OpenAI 兼容接口）
OPENAI_API_KEY=你的API Key
OPENAI_BASE_URL=https://api.minimaxi.com/v1
MODEL_ID=MiniMax-M2.7

# 智谱 AI（Embedding + OCR + Rerank）
ZHIPUAI_API_KEY=你的API Key
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 方案一：Chroma（本地嵌入，无需 Docker）

```bash
# 1. 构建知识库
python app/chroma/knowledge_build.py

# 2. 运行问答（任选一种）
python app/chroma/vector_search.py          # 纯向量检索
python app/chroma/hybrid_search.py          # 混合检索（完整 pipeline）
python app/chroma/vector_search_agent.py    # Agent 模式（纯向量）
python app/chroma/hybrid_search_agent.py    # Agent 模式（混合检索）
```

### 方案二：Milvus（需要 Docker）

```bash
# 1. 启动 Milvus 服务
cd app/milvus
docker compose up -d

# 2. 构建知识库
python app/milvus/knowledge_build.py

# 3. 运行问答（支持多轮对话）
python app/milvus/vector_search.py          # 纯向量检索
python app/milvus/hybrid_search.py          # 混合检索（多轮对话）
```

Milvus 管理界面：
- Attu（可视化管理）：http://localhost:3000
- MinIO 控制台：http://localhost:9001（minioadmin/minioadmin）
- WebUI：http://localhost:9091/webui

### 方案三：Qdrant（Docker 或本地文件）

```bash
# 1. 启动 Qdrant 服务
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 2. 构建知识库
python app/qdrant/knowledge_build.py

# 3. 运行问答
python app/qdrant/vector_search.py          # 纯向量检索（langchain_qdrant）
python app/qdrant/hybrid_search.py          # 混合检索（关键词补充 + RRF）
```

Qdrant Dashboard：http://localhost:6333/dashboard

## 知识库构建流程

```
1. Extract  ─  爬取帮助中心 25 个页面
                 │
                 ├─ 文本页面 → 直接提取正文
                 └─ 图片页面（< 50 字符）→ GLM-4V-Flash OCR 识别
                 
2. Transform ─  RecursiveCharacterTextSplitter 按中文标点切分
                 chunk_size=200, chunk_overlap=50
                 
3. Load      ─  ZhipuAI embedding-3 向量化，分批（64条/批）写入向量库
                 Milvus 额外自动生成 BM25 稀疏向量（内置 Function）
                 Qdrant 手动计算 BM25 稀疏向量并持久化词汇表/IDF
```

## 混合检索流程详解

### Chroma 方案：多查询扩展 + 手动 RRF + BGE Rerank

```
用户提问："盲推简历会怎样"
    │
    ▼
┌─────────────────────────────────────────┐
│  ① 多查询扩展（LLM）                     │
│  拆分为 2-3 个子查询：                     │
│    · "什么是盲推简历"                      │
│    · "盲推简历的处罚规则"                   │
│    · "推荐候选人的规范要求"                  │
└─────────────────────────────────────────┘
    │
    ▼  对每个子查询分别做双路检索
┌──────────────────┐   ┌──────────────────┐
│ Dense 向量检索     │   │ BM25 关键词检索    │
│ (ZhipuAI embed)  │   │ (jieba + rank_bm25)│
│ 每个子查询 top 10  │   │ 每个子查询 top 10   │
└────────┬─────────┘   └────────┬─────────┘
         │                      │
         ▼                      ▼
┌─────────────────────────────────────────┐
│  ② 合并去重 + RRF 融合排序（Python 端）    │
│  score = Σ weight / (k + rank)          │
│  取 top 6 候选                           │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  ③ BGE-Reranker 重排序（本地模型）         │
│  BAAI/bge-reranker-base                 │
│  用原始 query 对候选文档精排               │
│  取 top 2                               │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  ④ LLM 生成回答                          │
│  检索结果 + 用户问题 → MiniMax 生成        │
└─────────────────────────────────────────┘
```

### Milvus 方案：历史感知改写 + 服务端 RRF + 智谱 Rerank + 多轮对话

```
对话历史：[用户: "PM是什么", AI: "PM是禾蛙平台的工作人员..."]
用户追问："怎么联系他"
    │
    ▼
┌─────────────────────────────────────────┐
│  ① 历史感知查询改写（LLM）                 │
│  结合对话历史，解决指代词                    │
│  "怎么联系他" → "如何联系禾蛙平台的职位PM"   │
└─────────────────────────────────────────┘
    │
    ▼  改写后的 query 同时送入两路
┌──────────────────┐   ┌──────────────────┐
│ Dense 向量检索     │   │ Sparse BM25 检索  │
│ (ZhipuAI embed)  │   │ (Milvus 内置 BM25) │
│ COSINE top 10    │   │ 自动稀疏向量 top 10 │
└────────┬─────────┘   └────────┬─────────┘
         │                      │
         ▼                      ▼
┌─────────────────────────────────────────┐
│  ② RRF 融合排序（Milvus 服务端）           │
│  RRFRanker(k=60)                        │
│  取 top 6 候选                           │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  ③ 智谱 Rerank 重排序（API）              │
│  用原始 query 对候选文档精排               │
│  取 top 2                               │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  ④ LLM 生成回答（带对话历史）              │
│  system prompt + 最近 3 轮历史 + 检索结果  │
│  → MiniMax 生成连贯回答                   │
└─────────────────────────────────────────┘
    │
    ▼  记录本轮到 history（滑动窗口保留 3 轮）
```

### Qdrant 方案：关键词补充 + 服务端 Prefetch RRF

```
用户提问："PM咋联系"
    │
    ▼
┌─────────────────────────────────────────┐
│  ① 关键词补充（同义词词典）                 │
│  "PM" → 补充 "项目经理 工作人员 职位PM"     │
│  原始 query 用于 Dense，扩展 query 用于 BM25│
└─────────────────────────────────────────┘
    │
    ├─ 原始 query ──────┐   ┌── 扩展 query ─────┐
    ▼                    │   │                    ▼
┌──────────────────┐    │   │   ┌──────────────────┐
│ Dense 向量检索     │    │   │   │ BM25 稀疏向量检索  │
│ (ZhipuAI embed)  │    │   │   │ (jieba + 手动BM25) │
│ Prefetch top 10  │    │   │   │ Prefetch top 10   │
└────────┬─────────┘    │   │   └────────┬─────────┘
         │              │   │            │
         ▼              ▼   ▼            ▼
┌─────────────────────────────────────────┐
│  ② RRF 融合排序（Qdrant 服务端）           │
│  FusionQuery(fusion=Fusion.RRF)         │
│  取 top 6                               │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  ③ LLM 生成回答                          │
│  检索结果 + 用户问题 → MiniMax 生成        │
└─────────────────────────────────────────┘
```

## 查询优化策略对比

| 策略 | 文件 | 原理 | 优势 | 劣势 |
|------|------|------|------|------|
| **多查询扩展** | `query_expansion.py` | LLM 拆分为多个子查询，分别检索合并 | 覆盖面广，不易漏召回 | 延迟高（多次检索） |
| **LLM 查询改写** | `query_rewriter.py` | LLM 将口语化问题改写为精确查询 | 通用性强 | 一次 API 延迟 |
| **历史感知改写** | `contextual_rewriter.py` | 结合对话历史解决指代消歧 | 支持多轮追问 | 依赖历史质量 |
| **关键词补充** | `keyword_expansion.py` | 同义词词典扩展关键词 | 零延迟，增强 BM25 | 需要维护词典 |

## Reranker 重排序对比

| 方案 | 文件 | 模型 | 速度 | 成本 | 离线可用 |
|------|------|------|------|------|---------|
| **BGE-Reranker** | `reranker.py` | BAAI/bge-reranker-base（本地） | ~50-100ms | 免费 | 是 |
| **智谱 Rerank** | `zhipu_rerank.py` | 智谱 rerank（API） | ~500ms-2s | 免费 | 否 |

## 检索方式对比

### 纯向量检索（vector_search）

通过 Embedding 模型将查询转为向量，在向量空间中找最相似的文档。

- 优势：理解语义，"注销账号" 能匹配 "如何删除我的账户"
- 劣势：对精确关键词（如专有名词 "PM"）可能不够敏感

### 混合检索（hybrid_search）

同时使用向量语义检索和 BM25 关键词检索，通过 RRF 融合排序 + Reranker 精排。

- 优势：兼顾语义理解和关键词精确匹配，检索质量最高
- 完整 pipeline：查询优化 → Dense + BM25 → RRF → Rerank → LLM

### Agent 模式（*_agent）

使用 LangChain 的 `create_agent` + tool calling，由 LLM 自主决定是否调用检索工具。

- 优势：支持多步推理，LLM 可判断是否需要检索
- 劣势：依赖模型的 function calling 能力，部分模型可能不兼容

## 三套方案对比

| 特性 | Chroma | Milvus | Qdrant |
|------|--------|--------|--------|
| 部署复杂度 | 最低（纯本地） | 高（Docker Compose） | 中（Docker 单容器） |
| BM25 实现 | rank_bm25 库（Python 端） | 内置 BM25 Function（服务端） | 手动稀疏向量（入库时计算） |
| RRF 融合 | Python 端 | 服务端 | 服务端 |
| 查询优化 | 多查询扩展 | 历史感知改写 | 关键词补充 |
| Reranker | BGE 本地模型 | 智谱 API | - |
| 多轮对话 | - | 滑动窗口（3轮） | - |
| Agent 模式 | 支持 | 待实现 | 待实现 |
| 可视化工具 | 无 | Attu + MinIO | 内置 Dashboard |
| 适用场景 | 快速原型/小数据量 | 生产环境/大数据量 | 中等规模/灵活部署 |
