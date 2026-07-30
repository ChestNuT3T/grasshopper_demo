"""
=============================================================================
        RAG 系统 catalog 目录管理模块
        与其它模块的关系：
            - build_index.py 生成 CATALOG_PATH 和 LEDGER_PATH 文件
            - retrieval.py 在检索时调用 prefilter_by_catalog 做精确匹配预过滤
            - retrieval.py 在精确匹配命中后调用 get_chunk_by_id 加载完整文档
=============================================================================
"""
import json
from functools import lru_cache
from typing import Dict, Any, Optional, Tuple

from config.settings import (
    BATTERY_CATALOG_PATH,
    BATTERY_LEDGER_PATH,
    MANUAL_CATALOG_PATH,
    MANUAL_LEDGER_PATH,
)
from src.core.logger import logger

# =============================================================================
# 加载指定数据源（电池/手册）的 catalog 目录 JSON 文件，带 LRU 缓存
# =============================================================================
@lru_cache(maxsize=2)
def load_catalog(source: str) -> Dict[str, Any]:
    """
    加载指定数据源的目录（带缓存）。

    参数：
        source: 数据源类型，'battery' 或 'manual'

    返回：
        目录字典

    异常：
        ValueError: source 不是 'battery' 或 'manual'
        FileNotFoundError: 目录文件不存在
    """
    # 根据 source 确定 catalog 文件路径
    if source == "battery":
        catalog_path = BATTERY_CATALOG_PATH
    elif source == "manual":
        catalog_path = MANUAL_CATALOG_PATH
    else:
        raise ValueError(f"Invalid source: {source}. Must be 'battery' or 'manual'.")

    # 检查文件是否存在
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

    # 加载 JSON 文件
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    
    if source == "battery":
        count = len(catalog.get("batteries", []))
    else:
        count = len(catalog.get("chapters", []))
    logger.info(f"Loaded {source} catalog: {count} entries")
    return catalog

# =============================================================================
# 根据 keywords 在 catalog 中进行预匹配
# =============================================================================
def prefilter_by_catalog(keywords: dict, source: str) -> Tuple[Dict[str, Any], Optional[Dict]]:
    """
    根据关键词在目录中预过滤，生成 metadata_filter 和匹配条目。

    参数：
        keywords: 从 understanding 模块提取的关键词字典
                 包含 battery_plugin、operation、component 等字段
        source: "battery" 或 "manual"

    返回：
        (metadata_filter, catalog_match)
        metadata_filter: Chroma 过滤条件，如 {"battery_name": "Point"}
        catalog_match: 匹配到的目录条目，用于 Trace 记录，未匹配则返回 None
    """

    # 加载加载对应数据源的 catalog
    catalog = load_catalog(source)

    filter_cond = {}
    match = None

    # 对电池库进行匹配
    if source == "battery":
        # 从关键词中提取电池插件名称
        battery_plugin = keywords.get("battery_plugin", "")
        # 只有当 battery_plugin 非空且不为 "无" 时才进行匹配
        if battery_plugin and battery_plugin != "无":
            batteries = catalog.get("batteries", [])
            # 精确匹配（不区分大小写）
            for item in batteries:
                if item["battery_name"].lower() == battery_plugin.lower():
                    filter_cond = {"battery_name": item["battery_name"]}
                    match = item
                    break
            # 精确匹配失败，降级为包含匹配
            if not match:
                for item in batteries:
                    if battery_plugin.lower() in item["battery_name"].lower():
                        filter_cond = {"battery_name": item["battery_name"]}
                        match = item
                        break

    # 对手册库进行匹配
    else:
        # 从关键词中提取操作名称和组件名称
        operation = keywords.get("operation", "")
        component = keywords.get("component", "")
        search_terms = [t.lower() for t in [operation, component] if t and t != "无"]
        # 如果有搜索词，在章节标题中进行包含匹配
        if search_terms:
            chapters = catalog.get("chapters", [])
            for item in chapters:
                title = item.get("title", "").lower()
                # 只要章节标题包含任意一个搜索词即视为匹配
                if any(term in title for term in search_terms):
                    filter_cond = {"section": item["title"]}
                    match = item
                    break
    
    return filter_cond, match

# =============================================================================
# 根据 chunk_id 从底账（ledger）中加载完整的文档条目（当预匹配成功时，retrieval 模块会调用此函数加载完整文档）
# =============================================================================
def get_chunk_by_id(chunk_id: str) -> Optional[Dict[str, Any]]:
    """
    根据 chunk_id 从底账中获取完整的文档条目。

    参数：
        chunk_id: 文档块唯一标识，以 'battery:' 或 'manual:' 开头

    返回：
        文档条目字典（含 text 和所有 metadata），如果未找到则返回 None

    """
    if chunk_id.startswith("battery:"):
        ledger_path = BATTERY_LEDGER_PATH
    elif chunk_id.startswith("manual:"):
        ledger_path = MANUAL_LEDGER_PATH
    else:
        raise ValueError(f"Invalid chunk_id prefix: {chunk_id}")
    
    if not ledger_path.exists():
        logger.warning(f"Ledger file not found: {ledger_path}")
        return None
    
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                if entry.get("chunk_id") == chunk_id:
                    return entry
    
    logger.warning(f"Chunk not found: {chunk_id}")
    return None