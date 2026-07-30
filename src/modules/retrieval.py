"""
=============================================================================
        RAG 检索模块
        精确匹配（目录）→ 向量召回 → 重排序 → 返回最终文档。
=============================================================================
"""

import json
import os
import sys
import time
from typing import List, Tuple, Optional, Dict, Any
from http import HTTPStatus

import dashscope
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_community.query_constructors.chroma import ChromaTranslator

from src.modules.prompts import format_prompt
from src.modules.catalog import get_chunk_by_id, prefilter_by_catalog, load_catalog
from src.core.logger import logger, trace_logger
from src.core.trace import RetrievalTrace
from config.settings import RERANK_CONFIG, RETRIEVAL_CONFIG, get_retrieval_config

_RERANK_CACHE: Dict[str, List[Tuple[float, Document]]] = {}


# =============================================================================
# 全局重排序缓存
# =============================================================================
def _get_rerank_cache_key(query: str, docs: List[Document]) -> str:
    doc_signature = "|".join([doc.page_content[:100] for doc in docs])
    return f"{query[:200]}||{doc_signature}"


def _prune_rerank_cache() -> None:
    global _RERANK_CACHE
    max_size = RETRIEVAL_CONFIG.get("rerank_cache_max_size", 128)
    if len(_RERANK_CACHE) > max_size:
        oldest_keys = sorted(_RERANK_CACHE.keys())[:len(_RERANK_CACHE) - max_size]
        for key in oldest_keys:
            del _RERANK_CACHE[key]
        logger.debug(f"Rerank cache pruned, current size: {len(_RERANK_CACHE)}")

# =============================================================================
# 批量元数据提取（带 LLM 调用）
# =============================================================================
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2))
def extract_batch(batch_items: List[dict], llm_client) -> List[dict]:
    """
    核心函数：将一批（默认 10 条）电池 JSON 条目喂给 LLM，提取结构化元数据。

    参数：
        batch_items: 一批原始条目，每个包含 menuItem, title, texts 等字段。
        llm_client: 已初始化的 LLM 客户端。

    返回：
        每个条目Metadata元数据（doc_type, battery_name, menu_path, keywords, question_summary）列表。
    """
    entries_str = "\n".join([
        f"条目{i+1}：{json.dumps({k: v for k, v in item.items() if k in ['menuItem', 'title', 'texts']}, ensure_ascii=False)}"
        for i, item in enumerate(batch_items)
    ])

    # 构造 Prompt
    prompt_dict = format_prompt(
        "metadata_extraction",
        input_data=f"""
请为以下多个条目提取元数据，每个条目输出一行严格 JSON，不要序号前缀，保持原始顺序。
{entries_str}
"""
    )
    batch_prompt = prompt_dict["system"] + prompt_dict["user"]

    # 调用 LLM 进行元数据提取
    messages = [{"role": "user", "content": batch_prompt}]
    result = llm_client.invoke(messages)
    result_text = getattr(result, 'content', str(result))

    # 解析 LLM 返回的 JSON 结果
    lines = result_text.strip().split('\n')
    json_lines = [line.strip() for line in lines if line.strip().startswith('{')]

    # 校验输出结果数量
    if len(json_lines) != len(batch_items):
        raise ValueError(f"Batch size mismatch: expected {len(batch_items)} results, got {len(json_lines)}")

    metadata_list = []
    for line in json_lines:
        metadata_list.append(json.loads(line))

    # 合并原始数据和提取的元数据
    results = []
    for item, metadata in zip(batch_items, metadata_list):
        results.append({**item, **metadata})

    return results


def extract_metadata_for_json(json_path: str, llm_client) -> List[dict]:
    """
    带断点续传的批量元数据提取器。
    调用 extract_batch 函数，从JSON文件里面提取元数据 Metadata（`doc_type`、`battery_name`、`menu_path`、`keywords`、`question_summary`）
        `doc_type`：文档类型（"qa"、"battery"、"image_only"、"none"）
        `battery_name`：电池名称
        `menu_path`：目录
        `keywords`：关键词，≤3个
        `question_summary`：问题短句
    参数：
        json_path: JSON 文件路径。
        llm_client: LLM 客户端。
    返回：
        完整的 Metadata 元数据列表
    """
    BATCH_SIZE = 10  # 批量处理数量设置

    # 加载Json文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_count = len(data)  # Json文件总条目数量

    # 检查断点文件，如果存在则从断点处开始处理，如果不存在则创建于Json文件同级目录下，命名"checkpoint.json"
    json_dir = os.path.dirname(json_path)
    checkpoint_path = os.path.join(json_dir, "checkpoint.json") if json_dir else "checkpoint.json"

    results = []

    # 断点续传机制
    if os.path.exists(checkpoint_path):
        logger.info(f"Found checkpoint file, resuming from {checkpoint_path}")
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            results = json.load(f)

    start_idx = len(results)
    logger.info(f"Starting from index {start_idx}/{total_count}")

    # 批量处理，步长为BATCH_SIZE，调用 extract_batch 函数，通过LLM进行元数据提取
    for idx in range(start_idx, total_count, BATCH_SIZE):
        batch = data[idx:idx + BATCH_SIZE]
        batch_start = idx + 1
        batch_end = min(idx + BATCH_SIZE, total_count)

        try:
            batch_results = extract_batch(batch, llm_client)
            results.extend(batch_results)
            logger.info(f"Processed {batch_end}/{total_count} items")
        except Exception as e:
            RETRY_TIMES = 3
            error_msg = (
                f"\n{'='*60}\n"
                f"[FATAL] Batch {batch_start}-{batch_end} failed after {RETRY_TIMES} retries.\n"
                f"Error: {e}\n"
                f"Checkpoint saved up to item {start_idx}.\n"
                f"Please fix the issue and re-run the script to resume from the checkpoint.\n"
                f"{'='*60}"
            )
            logger.critical(error_msg)
            sys.exit(1)

        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False)

    # 全部处理完成，删除"checkpoint.json"断点文件
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info(f"Checkpoint file deleted: {checkpoint_path}")

    logger.info(f"Metadata extraction completed. Processed: {total_count}")
    return results

# =============================================================================
# 构建元数据缓存（将提取结果持久化，避免每次重新提取）
# =============================================================================
def build_metadata_cache(json_path: str, cache_path: str, llm_client) -> List[dict]:
    """
    如果缓存文件存在则直接加载，否则调用 extract_metadata_for_json 生成并保存到 cache_path。
    这个缓存主要用于加速后续的文档加载（避免每次启动都重新调用 LLM）。
    """
    if os.path.exists(cache_path):
        logger.info(f"Loading metadata cache from {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    logger.info("Creating metadata cache...")
    metadata_list = extract_metadata_for_json(json_path, llm_client)

    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)

    logger.info(f"Metadata cache saved to {cache_path}")
    return metadata_list


# =============================================================================
# 构建检索器
# =============================================================================
def load_battery_retriever(persist_dir: str, embeddings, llm, k: int):
    db = Chroma(embedding_function=embeddings, persist_directory=persist_dir, collection_name="battery_kb")
    return build_battery_retriever(db, llm, k)


def load_manual_retriever(persist_dir: str, embeddings, k: int):
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_dir, collection_name="manual_kb")
    return vectorstore.as_retriever(search_kwargs={"k": k})


# =============================================================================
# 构建电池检索器（SelfQueryRetriever）
# =============================================================================
def build_battery_retriever(battery_db: Chroma, llm, k: int) -> SelfQueryRetriever:
    metadata_field_info = [
        AttributeInfo(
            name="doc_type",
            description="文档类型标签。除非用户明确要求只查看某一类文档（如'只看电池'或'只看常见问题'），否则不要用此字段过滤。可选值：'battery'（电池说明）、'qa'（常见问题）、'image_only'（纯图片）",
            type="string",
        ),
        AttributeInfo(
            name="battery_name",
            description="电池的名称，可能包含英文原名和中文翻译，仅当 doc_type='battery' 时有值",
            type="string",
        ),
        AttributeInfo(
            name="menu_path",
            description="Grasshopper 菜单路径，如 'Params > Geometry > Rectangle'，分别表示主类别、父类别和电池本身",
            type="string",
        ),
        AttributeInfo(
            name="category",
            description="电池所属一级分类，如 Params、Maths、Surface 等",
            type="string",
        ),
        AttributeInfo(
            name="keywords",
            description="逗号分隔的关键词",
            type="string",
        ),
        AttributeInfo(
            name="question_summary",
            description="当 doc_type='qa' 时，提炼出的核心问题短句",
            type="string",
        ),
    ]
    document_content_description = "Grasshopper 电池说明；Grasshopper及Rhino常见问题解答。"
    retriever = SelfQueryRetriever.from_llm(
        llm,
        battery_db,
        document_content_description,
        metadata_field_info,
        search_kwargs={"k": k},
        structured_query_translator=ChromaTranslator(),
    )
    return retriever


# =============================================================================
# 重排序（Rerank）
# =============================================================================
def _rerank_docs(query: str, docs: List[Document], top_n: int) -> List[Document]:
    """
    对候选文档进行重排序，返回得分最高的 top_n 个文档。

    参数：
        query: 用户查询
        docs: 候选文档列表
        top_n: 最终返回的文档数量

    返回：
        重排后的文档列表
    """
    if len(docs) <= top_n:
        return docs

    rerank_model = RERANK_CONFIG.get("model", "qwen3-rerank")
    rerank_threshold = RERANK_CONFIG.get("threshold", 0.6)

    # 检查问题是否在缓存之中
    cache_key = _get_rerank_cache_key(query, docs)
    if cache_key in _RERANK_CACHE:
        logger.debug(f"Rerank cache hit for query: {query[:50]}...")
        cached_result = _RERANK_CACHE[cache_key]
        for score, doc in cached_result:
            doc.metadata["rerank_score"] = score
        return [doc for _, doc in cached_result]

    # 调用 Rerank API
    try:
        response = dashscope.TextReRank.call(
            model=rerank_model,
            query=query,
            documents=[doc.page_content for doc in docs],
            top_n=top_n,
            return_documents=False,
            api_key=os.getenv("DASHSCOPE_API_KEY")
        )

        if response.status_code == HTTPStatus.OK:
            results = response.output.get("results", [])
            scored_docs = []
            for result in results:
                index = result['index']
                score = result.get('relevance_score', 0)
                docs[index].metadata["rerank_score"] = score
                logger.info(f"Rerank score: {score:.4f} | {docs[index].page_content[:800]}...")
                if score >= rerank_threshold:
                    scored_docs.append((score, docs[index]))

            scored_docs.sort(key=lambda x: x[0], reverse=True)

            # 写入缓存
            _RERANK_CACHE[cache_key] = scored_docs
            _prune_rerank_cache()

            return [doc for _, doc in scored_docs]
        else:
            logger.warning(f"Rerank failed: {response.message}, fallback to top {top_n}")
            return docs[:top_n]

    except Exception as e:
        logger.warning(f"Rerank failed: {e}, fallback to top {top_n}")
        return docs[:top_n]


# =============================================================================
# 从底账加载文档
# =============================================================================
def load_documents_from_ledger_by_ids(chunk_ids: List[str]) -> List[Document]:
    """
    根据 chunk_id 从底账加载文档。

    参数：
        chunk_ids: chunk_id 列表

    返回：
        Document 列表，包含 text 和 metadata
    """
    documents = []
    for chunk_id in chunk_ids:
        entry = get_chunk_by_id(chunk_id)
        if entry:
            text = entry.get("text", "")
            metadata = {k: v for k, v in entry.items() if k != "text"}
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
    return documents


# =============================================================================
# 带 Trace 的检索
# =============================================================================
def retrieve_with_trace(
    query: str,
    metadata_filter: dict = None,
    catalog_match: Optional[Dict] = None,
    battery_top_k: int = 10,
    manual_top_k: int = 10,
    rerank_top_k: int = 5,
    candidate_multiplier: int = 2,
    fallback_min_docs: int = 3,
    source: str = "both",
    battery_retriever=None,
    manual_retriever=None,
) -> Tuple[List[Document], RetrievalTrace]:
    """
    执行完整检索流程，返回文档和追踪信息。

    参数说明：
        query: 用户查询
        metadata_filter: 元数据过滤条件（如 {"battery_name": "Circle"}）
        catalog_match: 目录匹配条目（用于追踪记录）
        battery_top_k: 电池库召回数量
        manual_top_k: 手册库召回数量
        rerank_top_k: 重排序后返回数量
        candidate_multiplier: 去重后候选池倍数（保留更多候选供重排序挑选）
        fallback_min_docs: 触发 Fallback 的最小文档数
        source: "battery_only" / "manual_only" / "both"
        battery_retriever: 电池检索器实例
        manual_retriever: 手册检索器实例

    返回：
        (documents, trace) 元组
    """

    start_time = time.time()
    trace = RetrievalTrace(query=query, rewritten_query=query)
    
    stage_times = {}
    
    if metadata_filter is None:
        metadata_filter = {}
    
    docs = []
    trace.metadata_filter = metadata_filter
    trace.catalog_match = catalog_match

    # -------------------------------------------------------------------------
    # 精确匹配（source 为 battery_only，且 metadata_filter 中有 battery_name）
    # -------------------------------------------------------------------------
    if source == "battery_only" and metadata_filter and "battery_name" in metadata_filter:
        prefilter_start = time.time()
        try:
            battery_name = metadata_filter.get("battery_name", "")
            if catalog_match and catalog_match.get("chunk_id"):
                matched_ids = [catalog_match["chunk_id"]]
                trace.catalog_match = catalog_match
            else:
                catalog = load_catalog("battery")
                batteries = catalog.get("batteries", [])
                matched_ids = []
                for item in batteries:
                    if item["battery_name"].lower() == battery_name.lower():
                        chunk_id = item.get("chunk_id", "")
                        if chunk_id:
                            matched_ids.append(chunk_id)
                            trace.catalog_match = item
                            break

            if matched_ids:
                docs = load_documents_from_ledger_by_ids(matched_ids)
                trace.match_mode = "exact"
                trace.stage_counts = {"battery": len(docs), "manual": 0}
                trace.results = [{"chunk_id": d.metadata.get("chunk_id", ""), "section": d.metadata.get("section", ""), "score": 1.0} for d in docs]
                
                stage_times["prefilter_ms"] = (time.time() - prefilter_start) * 1000
                stage_times["total_ms"] = (time.time() - start_time) * 1000
                trace.stage_times = stage_times
                
                trace.total_time_ms = stage_times["total_ms"]
                logger.info(f"Exact match retrieval completed: {len(docs)} docs, prefilter_ms={stage_times['prefilter_ms']:.2f}")
                return docs, trace
        except Exception as e:
            trace.errors.append(f"Exact match failed: {str(e)}")
            logger.warning(f"Exact match failed: {e}")

    # -------------------------------------------------------------------------
    # 向量检索，根据 source 参数决定从哪个知识库检索
    # -------------------------------------------------------------------------
    trace.match_mode = "vector"
    battery_docs = []
    manual_docs = []
    
    prefilter_start = time.time()
    vector_start = time.time()
    
    if source in ("battery_only", "both") and battery_retriever:
        try:
            battery_docs = battery_retriever.invoke(query)
        except Exception as e:
            trace.errors.append(f"Battery retrieval failed: {str(e)}")
            logger.warning(f"Battery retrieval failed: {e}")
    
    if source in ("manual_only", "both") and manual_retriever:
        try:
            manual_docs = manual_retriever.invoke(query)
        except Exception as e:
            trace.errors.append(f"Manual retrieval failed: {str(e)}")
            logger.warning(f"Manual retrieval failed: {e}")
    
    stage_times["prefilter_ms"] = (vector_start - prefilter_start) * 1000
    stage_times["vector_ms"] = (time.time() - vector_start) * 1000

    # -------------------------------------------------------------------------
    # 去重，两个知识库可能召回相同内容，按内容前 200 字符做哈希去重
    # -------------------------------------------------------------------------
    all_docs = []
    all_docs.extend(battery_docs)
    all_docs.extend(manual_docs)
    
    seen_content = set()
    unique_docs = []
    for doc in all_docs:
        content_hash = hash(doc.page_content[:200])
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_docs.append(doc)
    
    unique_docs = unique_docs[:candidate_multiplier * max(battery_top_k, manual_top_k)]

    # -------------------------------------------------------------------------
    # Fallback，去重后的结果 < fallback_min_docs，且 metadata_filter 非空
    # -------------------------------------------------------------------------
    if len(unique_docs) < fallback_min_docs and metadata_filter:
        trace.fallback_used = True
        logger.info(f"Low results ({len(unique_docs)}), falling back to no filter")
        fallback_docs = []
        if source in ("battery_only", "both") and battery_retriever:
            try:
                fallback_docs.extend(battery_retriever.invoke(query))
            except Exception:
                pass
        if source in ("manual_only", "both") and manual_retriever:
            try:
                fallback_docs.extend(manual_retriever.invoke(query))
            except Exception:
                pass

        # 对 fallback 结果也做去重
        seen = set()
        for doc in fallback_docs:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)
        
        unique_docs = unique_docs[:candidate_multiplier * max(battery_top_k, manual_top_k)]

    # -------------------------------------------------------------------------
    # 重排序，对候选文档进行精排，返回最相关的 rerank_top_k 个
    # -------------------------------------------------------------------------
    rerank_start = time.time()
    unique_docs = _rerank_docs(query, unique_docs, top_n=rerank_top_k)
    stage_times["rerank_ms"] = (time.time() - rerank_start) * 1000

    # -------------------------------------------------------------------------
    # 汇总耗时
    # -------------------------------------------------------------------------
    stage_times["total_ms"] = (time.time() - start_time) * 1000
    trace.stage_times = stage_times

    # -------------------------------------------------------------------------
    # 填充追踪信息
    # -------------------------------------------------------------------------
    trace.stage_counts = {"battery": len(battery_docs), "manual": len(manual_docs)}
    trace.results = []
    for d in unique_docs:
        result_entry = {
            "chunk_id": d.metadata.get("chunk_id", ""),
            "section": d.metadata.get("section", "") or d.metadata.get("battery_name", "") or d.metadata.get("chapter_title", ""),
            "score": d.metadata.get("rerank_score", 0.0),
            "source": d.metadata.get("source", ""),
            "doc_type": d.metadata.get("doc_type", "")
        }
        trace.results.append(result_entry)
    trace.total_time_ms = stage_times["total_ms"]

    # -------------------------------------------------------------------------
    # 日志记录
    # -------------------------------------------------------------------------
    trace_logger.info(json.dumps(trace.to_dict(), ensure_ascii=False))
    logger.info(f"Vector retrieval completed: {len(unique_docs)} docs, fallback_used: {trace.fallback_used}, "
                f"prefilter_ms={stage_times.get('prefilter_ms', 0):.2f}, "
                f"vector_ms={stage_times.get('vector_ms', 0):.2f}, "
                f"rerank_ms={stage_times.get('rerank_ms', 0):.2f}, "
                f"total_ms={stage_times.get('total_ms', 0):.2f}")
    return unique_docs, trace


# =============================================================================
# 检索参考文档（对外接口）
# =============================================================================
def retrieve_reference_docs(
    query: str,
    battery_retriever: SelfQueryRetriever,
    manual_retriever,
    retrieval_target: str = "both",
    keywords: Optional[Dict] = None,
    return_trace: bool = False,
    task_type: str = "knowledge_query"
) -> str | Tuple[str, RetrievalTrace]:
    """
    检索参考文档的主入口。

    参数：
        query: 用户查询
        battery_retriever: 电池检索器实例
        manual_retriever: 手册检索器实例
        retrieval_target: "battery_only" / "manual_only" / "both"
        keywords: 预提取的关键词（用于目录预匹配）
        return_trace: 是否返回追踪对象
        task_type: 任务类型，决定检索参数（knowledge_query / code_generation 等）

    返回：
        格式化后的参考文档字符串，或 (字符串, trace) 元组
    """
    start_time = time.time()

    # 根据 task_type 读取 retrieval 的配置
    retrieval_config = get_retrieval_config(task_type)
    battery_top_k = retrieval_config["battery_top_k"]
    manual_top_k = retrieval_config["manual_top_k"]
    rerank_top_k = retrieval_config["rerank_top_k"]
    candidate_multiplier = retrieval_config["candidate_multiplier"]
    fallback_min_docs = retrieval_config["fallback_min_docs"]
    
    logger.info(f"retrieve_reference_docs started - query: {query[:100]}..., target: {retrieval_target}, task_type: {task_type}, "
                f"battery_top_k: {battery_top_k}, manual_top_k: {manual_top_k}, rerank_top_k: {rerank_top_k}")

    metadata_filter = {}
    catalog_match = None

    # 目录预过滤
    if keywords:
        if retrieval_target in ("battery_only", "both"):
            battery_filter, battery_match = prefilter_by_catalog(keywords, "battery")
            if battery_match:
                metadata_filter = battery_filter
                catalog_match = battery_match
                logger.info(f"Battery catalog matched: {battery_match.get('battery_name')}")

        # 手册目录预过滤，仅记录Trace
        if not catalog_match and retrieval_target in ("manual_only", "both"):
            manual_filter, manual_match = prefilter_by_catalog(keywords, "manual")
            if manual_match:
                metadata_filter = manual_filter
                catalog_match = manual_match
                logger.info(f"Manual catalog matched: {manual_match.get('title')}")

    # 调用 retrieve_with_trace 函数，进行检索
    docs, trace = retrieve_with_trace(
        query=query,
        metadata_filter=metadata_filter,
        catalog_match=catalog_match,
        battery_top_k=battery_top_k,
        manual_top_k=manual_top_k,
        rerank_top_k=rerank_top_k,
        candidate_multiplier=candidate_multiplier,
        fallback_min_docs=fallback_min_docs,
        source=retrieval_target,
        battery_retriever=battery_retriever,
        manual_retriever=manual_retriever,
    )

    elapsed = time.time() - start_time
    logger.info(f"retrieve_reference_docs completed - unique_docs: {len(docs)}, match_mode: {trace.match_mode}, elapsed: {elapsed:.2f}s")

    if not docs:
        return ""

    # 格式化输出结果
    result_parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "")
        if source == "rhino_manual":
            source_label = "【Rhino 手册】"
        elif doc.metadata.get("doc_type") == "battery":
            source_label = "【电池知识库】"
        elif doc.metadata.get("doc_type") == "qa":
            source_label = "【常见问题】"
        else:
            source_label = "【参考文档】"

        title = (
            doc.metadata.get("battery_name", "") or
            doc.metadata.get("question_summary", "") or
            doc.metadata.get("chapter_title", "") or
            f"文档{i+1}"
        )
        page = doc.metadata.get("page", "") or doc.metadata.get("start_page", "")

        header = f"{source_label} {title}"
        if page:
            header += f" (第{page}页)"

        result_parts.append(f"{header}\n{doc.page_content}\n")

    result_str = "\n".join(result_parts)
    
    if return_trace:
        return result_str, trace
    return result_str
