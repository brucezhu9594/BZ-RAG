# BZ-RAG 知识库问答系统

基于 RAG（检索增强生成）架构的知识库问答系统，从帮助中心自动抓取文档，支持纯向量检索和混合检索两种方式，结合大模型生成回答。

## 架构概览

```
网页爬取 → 图片 OCR → 文本切分 → 向量化入库 → 检索 → LLM 生成回答
```

系统提供两套向量数据库实现（Chroma / Milvus），三种检索方式：

| 检索方式 | 实现 | 说明 |
|---------|------|------|
| 纯向量检索 | Chroma / Milvus | 基于语义相似度的稠密向量检索 |
| 手动混合检索 | Chroma | Dense + BM25 关键词检索，RRF 融合排序 |
| 原生混合检索 | Milvus | Dense + 内置 BM25 稀疏向量，RRF 融合排序，服务端完成 |

## 目录结构

```
BZ-RAG/
├── app/
│   ├── chroma/                  # Chroma 向量数据库实现
│   │   ├── knowledge_build.py   # 知识库构建（ETL + OCR）
│   │   ├── vector_search.py     # 纯向量检索 + RAG
│   │   ├── hybrid_search.py     # 混合检索 + RAG（手动 BM25 + RRF）
│   │   ├── bm25_index.py        # BM25 索引构建与缓存
│   │   ├── rrf.py               # RRF 融合排序算法
│   │   └── db/                  # Chroma 本地持久化数据
│   ├── milvus/                  # Milvus 向量数据库实现
│   │   ├── knowledge_build.py   # 知识库构建（ETL + OCR + BM25 稀疏向量）
│   │   ├── vector_search.py     # 纯向量检索 + RAG
│   │   └── hybrid_search.py     # 原生混合检索 + RAG
│   └── common/
│       └── image_ocr.py         # GLM-4V-Flash 图片文字识别
├── .env                         # 环境变量配置
├── .env.example                 # 环境变量模板
└── requirements.txt             # Python 依赖
```

## 环境依赖

- Python 3.10+
- Docker（Milvus 方案需要）

### 外部服务

| 服务 | 用途 | 配置项 |
|------|------|--------|
| MiniMax API | LLM 对话生成 | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `MODEL_ID` |
| 智谱 AI | 文本 Embedding（embedding-3） | `ZHIPUAI_API_KEY` |
| 智谱 AI | 图片 OCR（GLM-4V-Flash，免费） | `ZHIPUAI_API_KEY` |

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

# 智谱 AI（Embedding + OCR）
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

# 2. 运行问答（纯向量检索）
python app/chroma/vector_search.py

# 或运行混合检索
python app/chroma/hybrid_search.py
```

### 方案二：Milvus（需要 Docker）

```bash
# 1. 启动 Milvus 服务（etcd + MinIO + Milvus）
cd E:/docker/milvus
docker compose up -d

# 2. 构建知识库
python app/milvus/knowledge_build.py

# 3. 运行问答（纯向量检索）
python app/milvus/vector_search.py

# 或运行混合检索
python app/milvus/hybrid_search.py
```

## 知识库构建流程

```
1. Extract  ─  爬取帮助中心 25 个页面
                 │
                 ├─ 文本页面 → 直接提取正文
                 └─ 图片页面（< 50 字符）→ GLM-4V-Flash OCR 识别
                 
2. Transform ─  RecursiveCharacterTextSplitter 按中文标点切分
                 chunk_size=200, chunk_overlap=50
                 
3. Load      ─  ZhipuAI embedding-3 向量化，分批（64条/批）写入向量库
                 Milvus 方案额外自动生成 BM25 稀疏向量
```

## 检索方式对比

### 纯向量检索（vector_search）

通过 Embedding 模型将查询转为向量，在向量空间中找最相似的文档。

- 优势：理解语义，"注销账号" 能匹配 "如何删除我的账户"
- 劣势：对精确关键词（如专有名词 "PM"）可能不够敏感

### 混合检索（hybrid_search）

同时使用向量语义检索和 BM25 关键词检索，通过 RRF（Reciprocal Rank Fusion）融合排序。

- 优势：兼顾语义理解和关键词精确匹配，检索质量更高
- Chroma 方案：手动构建 BM25 索引 + Python 端 RRF 融合
- Milvus 方案：内置 BM25 Function + 服务端 RRF 融合，性能更优
