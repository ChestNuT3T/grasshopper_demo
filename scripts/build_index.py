"""
=============================================================================
        RAG 构建模块
        原始数据 → 底账(ledger) → 目录(catalog) → 向量库(Chroma)
=============================================================================
"""

import json
import os
import sys
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from pypdf import PdfReader
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import (
    BATTERY_LEDGER_PATH,
    BATTERY_CATALOG_PATH,
    MANUAL_LEDGER_PATH,
    MANUAL_CATALOG_PATH,
    BATTERY_PROCESSED_DIR,
    MANUAL_PROCESSED_DIR,
    CHROMA_DIR,
    JSON_PATH,
    PDF_PATH,
    DASHSCOPE_API_KEY,
)
from src.modules.retrieval import extract_metadata_for_json
from src.core.client import get_llm_client, get_embeddings
from src.core.logger import logger

# =============================================================================
# 从 JSONL 文件加载底账
# =============================================================================
def _load_ledger(path: Path) -> List[Dict[str, Any]]:
    """
    从 JSONL 文件加载底账。
    
    Args:
        path: JSONL 文件路径
    
    Returns:
        底账条目列表
    """
    ledger = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ledger.append(json.loads(line))
    return ledger

# =============================================================================
# 构建电池知识库底账（battery_ledger）
# =============================================================================
def build_battery_ledger(llm_client) -> List[Dict[str, Any]]:
    """
    构建电池知识库底账。
    
    从 JSON 数据源加载数据，调用 LLM 提取元数据Metadata（battery_name、keywords、question_summary），
    结合代码自动生成的字段（doc_type、source、menu_path、category、text），生成完整的底账条目。
    
    Args:
        llm_client: LLM 客户端实例，用于调用 extract_metadata_for_json 提取元数据
    
    Returns:
        底账条目列表，每个条目包含完整的元数据字段
    """
    # 创建电池库处理目录（如果不存在）
    BATTERY_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # 如果底账文件已存在，直接加载返回（支持断点续建）
    if BATTERY_LEDGER_PATH.exists():
        logger.info(f"加载已存在的电池底账: {BATTERY_LEDGER_PATH}")
        return _load_ledger(BATTERY_LEDGER_PATH)
    
    logger.info("开始构建电池底账...")
    
    # 生成文档 ID（基于 JSON 文件内容的 MD5 哈希，取前12位）
    with open(JSON_PATH, 'rb') as f:
        doc_id = hashlib.md5(f.read()).hexdigest()[:12]
    
    # 调用 extract_metadata_for_json 函数，使用 LLM 批量提取元数据（含断点续提机制）
    metadata_list = extract_metadata_for_json(JSON_PATH, llm_client)
    
    ledger = []
    for idx, item in enumerate(metadata_list):
        # 获取文档类型，跳过 "doc_type" = "image_only", "none" 的无效文档
        doc_type = item.get("doc_type", "")
        if doc_type in ("image_only", "none"):
            continue
        
        # 提取菜单路径"menu_path"和正文内容"texts"
        menu_path = item.get("menu_path", "")
        texts = item.get("texts", [])
        text = " ".join(texts)
        
        # 跳过空内容条目
        if not text.strip():
            continue
        
        # 从菜单路径提取一级分类（如 Params、Vector 等）
        category = menu_path.split(" > ")[0] if menu_path else ""
        
        # 构建完整的底账条目
        entry = {
            "chunk_id": f"battery:{doc_id}:{hashlib.md5(menu_path.encode()).hexdigest()[:8]}:000",
            "doc_id": doc_id,
            "text": text,
            "section": menu_path,       # 写入chrome向量库
            "doc_type": doc_type,
            "battery_name": item.get("battery_name", ""),
            "menu_path": menu_path,     # 写入catalog目录
            "category": category,
            "keywords": item.get("keywords", ""),
            "question_summary": item.get("question_summary", ""),
            "source": "battery" if doc_type == "battery" else "qa",
            "chunk_index": 0,
            "is_complete": True
        }
        ledger.append(entry)
    
    # 逐行写入 JSONL 文件
    with open(BATTERY_LEDGER_PATH, "w", encoding="utf-8") as f:
        for entry in ledger:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    logger.info(f"电池底账构建完成: {len(ledger)} 条目")
    return ledger

# =============================================================================
# 构建电池知识库目录（battery_catalog）
# =============================================================================
def build_battery_catalog(ledger: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    构建电池知识库目录。
    
    从底账中聚合提取电池和QA信息，生成符合架构优化的目录结构，
    包含 battery_count、qa_count、categories、batteries、qa_list 等字段。
    
    Args:
        ledger: 电池底账条目列表
    
    Returns:
        目录字典，包含完整的电池和QA统计信息
    """
    battery_map = {}    # 存放电池信息（去重并统计电池数量）
    qa_list = []        # 存放QA信息
    categories = set()
    
    for entry in ledger:
        doc_type = entry.get("doc_type", "")
        if doc_type == "battery":
            name = entry.get("battery_name", "")
            if not name:
                continue
            if name not in battery_map:
                battery_map[name] = {
                    "battery_name": name,
                    "menu_path": entry.get("menu_path", ""),
                    "category": entry.get("category", ""),
                    "keywords": entry.get("keywords", "").split(",") if entry.get("keywords") else [],
                    "chunk_count": 0,
                    "text_preview": entry.get("text", "")[:100] + "...",
                    "chunk_id": entry.get("chunk_id", "")
                }
            battery_map[name]["chunk_count"] += 1
            categories.add(entry.get("category", ""))
        elif doc_type == "qa":
            qa_list.append({
                "question_summary": entry.get("question_summary", ""),
                "menu_path": entry.get("menu_path", ""),
                "chunk_id": entry.get("chunk_id", "")
            })
    
    catalog = {
        "battery_count": len(battery_map),
        "qa_count": len(qa_list),
        "categories": sorted(list(categories)),
        "batteries": list(battery_map.values()),
        "qa_list": qa_list
    }
    
    with open(BATTERY_CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    
    logger.info(f"电池目录构建完成: {len(battery_map)} 个电池, {len(qa_list)} 个 QA")
    return catalog

# =============================================================================
# PDF 书签大纲提取与切分（新增核心功能）
# =============================================================================
def extract_bookmarks_structure(pdf_path: str) -> List[Dict[str, Any]]:
    """
    从 PDF 书签大纲提取层级结构。
    
    pypdf 的 outline 结构为嵌套列表：Destination 对象和子列表交替出现，
    子列表代表下一层级的书签。
    
    Args:
        pdf_path: PDF 文件路径
        
    Returns:
        书签树列表，每个节点包含 title、page、level、children
    """
    reader = PdfReader(pdf_path)
    if not reader.outline:
        raise ValueError("PDF 文件不包含书签大纲")

    # 递归遍历书签列表
    # items: 当前层级的书签列表（可能混合 Destination 和子列表）
    # level: 当前层级深度（根为1）
    def traverse(items, level=1):
        result = []
        i = 0
        while i < len(items):
            item = items[i]
            # 情况1：当前元素是子列表 → 直接展开（它是上一级书签的子节点，但可能在同级中穿插）
            if isinstance(item, list):
                result.extend(traverse(item, level))
                i += 1
            else:
            # 情况2：当前元素是 Destination 对象 → 构建书签节点
                page = reader.get_destination_page_number(item)
                node = {"title": item.title, "page": page, "level": level, "children": []}
                i += 1
                # 检查下一个元素是否为子列表（属于当前书签的子节点）
                if i < len(items) and isinstance(items[i], list):
                    node["children"] = traverse(items[i], level + 1)
                    i += 1
                result.append(node)
        return result
    
    return traverse(reader.outline)


def flatten_bookmarks(tree: List[Dict], parent_path: str = "") -> List[Dict]:
    """
    将书签树展平为列表，每个条目包含完整层级路径。
    
    Args:
        tree: 书签树列表
        parent_path: 父级路径
        
    Returns:
        展平后的书签列表，每个条目包含 title、page、level、path
    """
    result = []
    for node in tree:
        path = f"{parent_path} > {node['title']}" if parent_path else node['title']
        flat = {"title": node["title"], "page": node["page"], "level": node["level"], "path": path}
        result.append(flat)
        if node.get("children"):
            result.extend(flatten_bookmarks(node["children"], path))
    return result


def split_by_bookmarks(pdf_path: str, doc_id: str, flat_bookmarks: List[Dict]) -> List[Dict]:
    """
    按书签边界切分 PDF 内容，每个书签生成一个 chunk。
    
    Args:
        pdf_path: PDF 文件路径
        doc_id: 文档 ID
        flat_bookmarks: 展平后的书签列表
        
    Returns:
        切分后的 chunk 列表
    """
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    chunks = []
    
    for i, entry in enumerate(flat_bookmarks):
        start = entry["page"]
        end = flat_bookmarks[i + 1]["page"] - 1 if i + 1 < len(flat_bookmarks) else total_pages
        if start > end:
            end = start

        # 提取页面文本
        text_parts = []
        for p in range(start - 1, min(end, total_pages)):
            text_parts.append(reader.pages[p].extract_text() or "")
        
        full = "\n".join(text_parts).strip()
        if not full:
            continue
        
        chunk_hash = hashlib.md5(entry["path"].encode()).hexdigest()[:8]
        
        chunks.append({
            "chunk_id": f"manual:{doc_id}:{chunk_hash}:{i:03d}",
            "doc_id": doc_id,
            "text": full,
            "section": entry["path"],
            "doc_type": "manual",
            "chapter_title": entry["title"],
            "chapter_level": entry["level"],
            "source": "rhino_manual",
            "chunk_index": i,
            "is_complete": True,
            "start_page": start,
            "end_page": end,
        })
    
    return chunks


# =============================================================================
# 构建手册知识库底账（manual_ledger）
# =============================================================================
def build_manual_ledger() -> List[Dict[str, Any]]:
    """
    构建手册知识库底账（基于 PDF 书签大纲驱动）。
    
    使用 pypdf 的 reader.outline 提取书签层级，按书签边界精确切分内容，

    Returns:
        底账条目列表，每个条目包含章节信息和正文内容
    """
    MANUAL_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 如果完整底账已存在，直接加载
    if MANUAL_LEDGER_PATH.exists():
        logger.info(f"加载已存在的手册底账: {MANUAL_LEDGER_PATH}")
        return _load_ledger(MANUAL_LEDGER_PATH)
    
    checkpoint = MANUAL_PROCESSED_DIR / "manual_checkpoint.jsonl"
    if checkpoint.exists():
        logger.warning(f"发现断点文件 {checkpoint}，将删除并重新构建（确保数据完整性）")
        os.remove(checkpoint)
    
    logger.info("开始从 PDF 书签大纲构建手册底账...")
    
    # 生成文档 ID（基于 PDF 文件内容的 MD5 哈希，取前12位）
    with open(PDF_PATH, 'rb') as f:
        doc_id = hashlib.md5(f.read()).hexdigest()[:12]

    # 调用 extract_bookmarks_structure 提取书签，调用 flatten_bookmarks 展平书签
    tree = extract_bookmarks_structure(PDF_PATH)
    flat = flatten_bookmarks(tree)

    # 打印书签统计信息（方便调试）
    level_dist = {l: sum(1 for b in flat if b['level'] == l) for l in set(b['level'] for b in flat)}
    logger.info(f"书签总数: {len(flat)}, 层级分布: {level_dist}")

    # 按书签边界切分内容
    chunks = split_by_bookmarks(PDF_PATH, doc_id, flat)
    
    with open(MANUAL_LEDGER_PATH, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    
    logger.info(f"手册底账构建完成: {len(chunks)} 个 chunks")
    return chunks

# =============================================================================
# 构建手册知识库目录（manual_catalog）
# =============================================================================
def build_manual_catalog(ledger: List[Dict[str, Any]]) -> Dict[str, Any]:
    """从手册底账中聚合出章节目录，包含每个章节的起止页、关键词、包含的 chunk_ids"""
    chapter_map = {}
    
    for entry in ledger:
        title = entry.get("chapter_title", "")
        if not title:
            continue
        
        if title not in chapter_map:
            chapter_map[title] = {
                "title": title,
                "level": entry.get("chapter_level", 1),
                "page_start": entry.get("start_page", 1),
                "page_end": entry.get("end_page", 1),
                "chunk_ids": []
            }
        chapter_map[title]["chunk_ids"].append(entry.get("chunk_id", ""))
    
    # 生成章节列表
    chapters = []
    for title, data in chapter_map.items():
        chapters.append({
            "title": data["title"],
            "level": data["level"],
            "page_range": f"{data['page_start']}-{data['page_end']}",
            "chunk_ids": data["chunk_ids"]
        })
    
    catalog = {
        "chapter_count": len(chapters),
        "chapters": chapters
    }
    
    with open(MANUAL_CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    
    logger.info(f"手册目录构建完成: {len(chapters)} 个章节")
    return catalog

# =============================================================================
# 从底账构建 Chroma 向量库
# =============================================================================
def build_vectorstore_from_ledger(
    battery_ledger: List[Dict[str, Any]],
    manual_ledger: List[Dict[str, Any]],
    embeddings,
) -> Dict[str, Chroma]:
    """
    从底账构建 Chroma 向量库。
    
    将电池库和手册库的底账条目转换为 LangChain Document 对象，
    使用 text-embedding-v4 模型生成向量，分别写入两个独立的 Chroma collection。
    
    Args:
        battery_ledger: 电池底账条目列表
        manual_ledger: 手册底账条目列表
        embeddings: 嵌入模型实例（DashScopeEmbeddings）
    
    Returns:
        包含两个向量库的字典：battery_kb 和 manual_kb
    """
    # 创建 Chroma 向量库存储目录（如果不存在）
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    
    stores = {}
    
    # 构建电池知识库向量库（collection: battery_kb）
    stores["battery_kb"] = _build_single_vectorstore(
        battery_ledger,
        "battery_kb",
        embeddings,
    )
    
    # 构建手册知识库向量库（collection: manual_kb）
    stores["manual_kb"] = _build_single_vectorstore(
        manual_ledger,
        "manual_kb",
        embeddings,
    )
    
    return stores


def _build_single_vectorstore(
    ledger: List[Dict[str, Any]],
    collection_name: str,
    embeddings,
) -> Chroma:
    """
    构建单个 Chroma 向量库。
    
    将底账条目转换为 LangChain Document 对象，生成向量并写入指定的 Chroma collection。
    如果该 collection 已存在且包含文档，则直接返回已有的向量库（支持增量构建）。
    
    Args:
        ledger: 底账条目列表
        collection_name: Chroma collection 名称
        embeddings: 嵌入模型实例
    
    Returns:
        Chroma 向量库实例
    """
    # 尝试加载已存在的向量库（支持增量构建）
    try:
        existing_db = Chroma(
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
            collection_name=collection_name,
        )
        count = existing_db._collection.count()
        if count > 0:
            logger.info(f"加载已存在的 {collection_name}，共 {count} 个文档")
            return existing_db
    except Exception as e:
        logger.info(f"{collection_name} 不存在，将新建: {e}")
    
    logger.info(f"开始构建 {collection_name}...")
    
    # 将底账条目转换为 LangChain Document 对象
    documents = []
    for entry in ledger:
        source = entry.get("source", "")
        # 构建基础 metadata 字典（包含所有索引时需要的元数据字段）
        metadata = {
            "chunk_id": entry["chunk_id"],
            "doc_type": entry.get("doc_type", ""),
            "source": source,
            "keywords": entry.get("keywords", ""),
        }
        
        # 电池库特有字段
        if source in ("battery", "qa"):
            metadata["battery_name"] = entry.get("battery_name", "")
            metadata["menu_path"] = entry.get("menu_path", "")
            metadata["category"] = entry.get("category", "")
            metadata["question_summary"] = entry.get("question_summary", "")
        
        # 手册库特有字段
        elif source == "rhino_manual":
            metadata["chapter_title"] = entry.get("chapter_title", "")
            metadata["section"] = entry.get("section", "")
            metadata["start_page"] = entry.get("start_page", "")
            metadata["end_page"] = entry.get("end_page", "")
        
        # 创建 Document 对象（page_content 为正文，metadata 为元数据）
        doc = Document(page_content=entry["text"], metadata=metadata)
        documents.append(doc)
    
    logger.info(f"正在创建 {collection_name}，共 {len(documents)} 个文档...")
    
    # 使用 Chroma.from_documents 将文档写入向量库（自动生成向量）
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=collection_name,
    )
    
    logger.info(f"{collection_name} 构建完成")
    return db


def main():
    """
    主入口函数：执行 RAG 索引构建的完整流程。
    
    按顺序执行以下步骤：
    1. 初始化 LLM 客户端和嵌入模型
    2. 构建电池知识库底账（调用 LLM 提取元数据）
    3. 构建电池知识库目录（聚合底账信息）
    4. 构建手册知识库底账（解析 PDF，按章节切分）
    5. 构建手册知识库目录（聚合底账信息）
    6. 构建 Chroma 向量库（将底账转换为向量）
    """
    logger.info("=" * 60)
    logger.info("开始执行 RAG 索引构建流程")
    logger.info("=" * 60)
    
    try:
        # 初始化 LLM 客户端（用于提取元数据）和嵌入模型（用于生成向量）
        llm_client = get_llm_client()
        embeddings = get_embeddings()
        
        # 步骤1：构建电池知识库底账
        logger.info("\n--- 构建电池知识库底账 ---")
        battery_ledger = build_battery_ledger(llm_client)
        
        # 步骤2：构建电池知识库目录
        logger.info("\n--- 构建电池知识库目录 ---")
        build_battery_catalog(battery_ledger)
        
        # 步骤3：构建手册知识库底账
        logger.info("\n--- 构建手册知识库底账 ---")
        manual_ledger = build_manual_ledger()
        
        # 步骤4：构建手册知识库目录
        logger.info("\n--- 构建手册知识库目录 ---")
        build_manual_catalog(manual_ledger)
        
        # 步骤5：构建 Chroma 向量库
        logger.info("\n--- 构建向量库 ---")
        stores = build_vectorstore_from_ledger(battery_ledger, manual_ledger, embeddings)
        
        # 构建完成
        logger.info("\n" + "=" * 60)
        logger.info("RAG 索引构建流程执行成功")
        logger.info("=" * 60)
        
    except Exception as e:
        # 捕获并记录所有异常，输出详细堆栈信息
        logger.error(f"构建流程失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()