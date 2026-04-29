"""RAG 评测脚本：评估检索质量和生成质量，支持多方案对比。"""

import json
import os
import pathlib
import re
import sys
import time

PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DATASET_PATH = pathlib.Path(__file__).parent / "test_dataset.json"
RESULT_DIR = pathlib.Path(__file__).parent / "results"
RESULT_DIR.mkdir(exist_ok=True)

MODEL = os.environ["MODEL_ID"]


# ========== 检索方案注册 ==========

CHROMA_DIR = os.path.join(PROJECT_ROOT, "app", "chroma")


def chroma_vector_retrieve(query: str):
    saved_cwd = os.getcwd()
    os.chdir(CHROMA_DIR)
    try:
        from app.chroma.vector_search import _retrieve_for_query

        serialized, docs = _retrieve_for_query(query)
        return [{"content": d.page_content, "source": d.metadata.get("source", "")} for d in docs]
    finally:
        os.chdir(saved_cwd)


def chroma_hybrid_retrieve(query: str):
    saved_cwd = os.getcwd()
    os.chdir(CHROMA_DIR)
    try:
        from app.chroma.hybrid_search import _retrieve

        serialized, docs = _retrieve(query)
        return [{"content": d.page_content, "source": d.metadata.get("source", "")} for d in docs]
    finally:
        os.chdir(saved_cwd)


def milvus_vector_retrieve(query: str):
    from app.milvus.vector_search import _retrieve_for_query

    serialized, results = _retrieve_for_query(query)
    return [
        {"content": r["entity"].get("text", ""), "source": r["entity"].get("source", "")}
        for r in results
    ]


def milvus_hybrid_retrieve(query: str):
    from app.milvus.hybrid_search import _retrieve

    serialized, docs = _retrieve(query)
    return [{"content": d.page_content, "source": d.metadata.get("source", "")} for d in docs]


RETRIEVE_METHODS = {
    "chroma_vector": chroma_vector_retrieve,
    "chroma_hybrid": chroma_hybrid_retrieve,
    "milvus_vector": milvus_vector_retrieve,
    "milvus_hybrid": milvus_hybrid_retrieve,
}


# ========== 评测指标 ==========


def hit_rate(retrieved: list[dict], expected_source: str) -> float:
    """命中率：检索结果中是否包含期望来源的文档。"""
    for doc in retrieved:
        if expected_source in doc.get("source", ""):
            return 1.0
    return 0.0


def mrr(retrieved: list[dict], expected_source: str) -> float:
    """MRR：期望文档排在第几位（倒数排名）。"""
    for i, doc in enumerate(retrieved):
        if expected_source in doc.get("source", ""):
            return 1.0 / (i + 1)
    return 0.0


def _extract_score(text: str) -> float:
    """从 LLM 输出中提取 0-1 之间的评分数字。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # 尝试直接转换
    try:
        val = float(text)
        if 0 <= val <= 1:
            return val
    except ValueError:
        pass
    # 从文本中提取 0.0/0.5/1.0 等数字
    matches = re.findall(r"\b([01](?:\.\d)?)\b", text)
    if matches:
        return float(matches[-1])
    # 根据关键词推断
    if any(w in text for w in ["完全忠实", "高度相关", "完全相关"]):
        return 1.0
    if any(w in text for w in ["编造", "无关", "答非所问"]):
        return 0.0
    return 0.5


def faithfulness_score(answer: str, context: str) -> float:
    """忠实度：使用 LLM 评估回答是否忠于检索结果（0-1 分）。"""
    try:
        llm = ChatOpenAI(model=MODEL, temperature=0, request_timeout=30)
        prompt = (
            "请评估以下「回答」是否忠实于「参考内容」，即回答中的信息是否都能从参考内容中找到依据。\n"
            "评分标准：1.0=完全忠实，0.5=部分忠实有少量推测，0.0=包含编造内容\n"
            "只输出一个数字（0.0/0.5/1.0），不要解释。\n\n"
            f"参考内容：{context[:500]}\n\n回答：{answer[:500]}"
        )
        msg = llm.invoke([{"role": "user", "content": prompt}])
        return _extract_score(msg.content or "")
    except Exception as e:
        print(f"  [忠实度评分失败: {e}]")
        return 0.5


def relevance_score(answer: str, question: str) -> float:
    """相关性：使用 LLM 评估回答是否切题（0-1 分）。"""
    try:
        llm = ChatOpenAI(model=MODEL, temperature=0, request_timeout=30)
        prompt = (
            "请评估以下「回答」与「问题」的相关性，即回答是否针对问题进行了有效回答。\n"
            "评分标准：1.0=高度相关且有效回答，0.5=部分相关，0.0=完全无关或答非所问\n"
            "只输出一个数字（0.0/0.5/1.0），不要解释。\n\n"
            f"问题：{question}\n\n回答：{answer[:500]}"
        )
        msg = llm.invoke([{"role": "user", "content": prompt}])
        return _extract_score(msg.content or "")
    except Exception as e:
        print(f"  [相关性评分失败: {e}]")
        return 0.5


# ========== 生成回答 ==========


def generate_answer(query: str, context: str) -> str:
    """基于检索结果生成回答。"""
    try:
        llm = ChatOpenAI(model=MODEL, temperature=0.7, request_timeout=60)
        system_prompt = (
            "你是一个知识库检索助手。"
            "下面「检索结果」来自知识库片段，请仅依据这些内容回答用户问题。"
            "如果检索结果不足以回答，请明确说明知识库中没有相关信息，不要编造。"
            f"\n\n--- 检索结果 ---\n{context}"
        )
        msg = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]
        )
        return msg.content or ""
    except Exception as e:
        print(f"  [生成回答失败: {e}]")
        return ""


# ========== 主评测流程 ==========


def run_evaluation(method_name: str, retrieve_fn, dataset: list[dict]) -> dict:
    """对单个检索方案跑完整评测。"""
    print(f"\n{'='*60}")
    print(f"评测方案: {method_name}")
    print(f"{'='*60}")

    results = []
    total_hit = 0
    total_mrr = 0
    total_faithfulness = 0
    total_relevance = 0
    total_latency = 0

    for i, item in enumerate(dataset):
        question = item["question"]
        expected_source = item["expected_source"]
        print(f"\n[{i+1}/{len(dataset)}] {question}")

        # 检索
        start = time.time()
        try:
            retrieved = retrieve_fn(question)
        except Exception as e:
            print(f"  检索失败: {e}")
            results.append({"question": question, "error": str(e)})
            continue
        retrieve_time = time.time() - start

        # 检索指标
        hr = hit_rate(retrieved, expected_source)
        mr = mrr(retrieved, expected_source)
        total_hit += hr
        total_mrr += mr

        # 拼接上下文
        context = "\n\n".join(
            f"Source: {d.get('source', '')}\nContent: {d.get('content', '')}" for d in retrieved
        )

        # 生成回答
        start = time.time()
        answer = generate_answer(question, context)
        generate_time = time.time() - start
        latency = retrieve_time + generate_time
        total_latency += latency

        # 生成质量指标
        faith = faithfulness_score(answer, context)
        relev = relevance_score(answer, question)
        total_faithfulness += faith
        total_relevance += relev

        print(
            f"  命中={hr:.0f}  MRR={mr:.2f}  忠实度={faith:.1f}  相关性={relev:.1f}  耗时={latency:.1f}s"
        )

        results.append(
            {
                "question": question,
                "expected_source": expected_source,
                "retrieved_sources": [d.get("source", "") for d in retrieved],
                "answer": answer[:200],
                "hit_rate": hr,
                "mrr": mr,
                "faithfulness": faith,
                "relevance": relev,
                "retrieve_time": round(retrieve_time, 2),
                "generate_time": round(generate_time, 2),
            }
        )

    n = len([r for r in results if "error" not in r])
    summary = {
        "method": method_name,
        "total_questions": len(dataset),
        "successful": n,
        "avg_hit_rate": round(total_hit / n, 4) if n else 0,
        "avg_mrr": round(total_mrr / n, 4) if n else 0,
        "avg_faithfulness": round(total_faithfulness / n, 4) if n else 0,
        "avg_relevance": round(total_relevance / n, 4) if n else 0,
        "avg_latency": round(total_latency / n, 2) if n else 0,
    }

    print(f"\n--- {method_name} 汇总 ---")
    print(f"  命中率:   {summary['avg_hit_rate']:.2%}")
    print(f"  MRR:     {summary['avg_mrr']:.4f}")
    print(f"  忠实度:   {summary['avg_faithfulness']:.2%}")
    print(f"  相关性:   {summary['avg_relevance']:.2%}")
    print(f"  平均耗时: {summary['avg_latency']:.1f}s")

    return {"summary": summary, "details": results}


def main():
    # 加载测试数据集
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    # 选择要评测的方案（命令行参数或全部）
    methods_to_run = sys.argv[1:] if len(sys.argv) > 1 else list(RETRIEVE_METHODS.keys())

    all_summaries = []
    for method_name in methods_to_run:
        if method_name not in RETRIEVE_METHODS:
            print(f"未知方案: {method_name}，可选: {list(RETRIEVE_METHODS.keys())}")
            continue

        result = run_evaluation(method_name, RETRIEVE_METHODS[method_name], dataset)
        all_summaries.append(result["summary"])

        # 保存详细结果
        output_path = RESULT_DIR / f"{method_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  详细结果已保存: {output_path}")

    # 打印对比表
    if len(all_summaries) > 1:
        print(f"\n{'='*60}")
        print("方案对比")
        print(f"{'='*60}")
        header = f"{'方案':<20} {'命中率':>8} {'MRR':>8} {'忠实度':>8} {'相关性':>8} {'耗时':>8}"
        print(header)
        print("-" * len(header))
        for s in all_summaries:
            print(
                f"{s['method']:<20} {s['avg_hit_rate']:>7.2%} {s['avg_mrr']:>8.4f} "
                f"{s['avg_faithfulness']:>7.2%} {s['avg_relevance']:>7.2%} {s['avg_latency']:>7.1f}s"
            )


if __name__ == "__main__":
    main()
