"""
=============================================================================
        客户端模块：提供 LLM 客户端和 Embedding 客户端的创建和管理功能
=============================================================================
"""

import os
import time
import threading
from collections import OrderedDict
from typing import Dict, Optional, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.embeddings.dashscope import DashScopeEmbeddings

from src.core.logger import logger

# =============================================================================
# 创建 DeepSeek API 客户端
# =============================================================================
def create_deepseek_client(
    model: str = "deepseek-v4-pro",
    max_tokens: int = None,
    streaming: bool = True,
    api_key: Optional[str] = None,
    base_url: str = "https://api.deepseek.com",
    timeout: int = 30,
    enable_reasoning: bool = True,
    reasoning_effort: str = "high",
    json_mode: bool = False
) -> ChatOpenAI:
    """
    创建 DeepSeek API 客户端（基于 langchain ChatOpenAI 封装）。

    Args:
        model: 模型名称，默认 deepseek-v4-pro
        max_tokens: 最大输出 token 数
        streaming: 是否启用流式输出
        api_key: API Key，默认从 settings 读取
        base_url: API base URL
        timeout: 请求超时秒数
        enable_reasoning: 是否启用 reasoning 模式（仅流式生效）
        reasoning_effort: reasoning 努力程度
        json_mode: 是否启用 JSON Output 模式（response_format=json_object）。
                   启用后 prompt 必须含 "json" 字样和格式样例，否则 API 会报错。
                   注意：官方有概率返回空 content，需配合重试使用。

    Returns:
        ChatOpenAI 客户端实例

    Raises:
        ValueError: DEEPSEEK_API_KEY 未设置
    """
    from config.settings import DEEPSEEK_API_KEY
    api_key = DEEPSEEK_API_KEY

    # 检查 API Key 是否存在
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 环境变量未设置")

    # 构建 model_kwargs，仅在流式模式下设置 stream_options
    model_kwargs = {}
    if streaming:
        model_kwargs["stream_options"] = {"include_usage": True}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    # 动态构建 ChatOpenAI 参数
    client_kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "streaming": streaming,
        "api_key": api_key,
        "base_url": base_url,
        "callbacks": None,
        "model_kwargs": model_kwargs,
        "timeout": timeout
    }
    
    # 启用推理模式时添加 extra_body 和 reasoning_effort
    if enable_reasoning and streaming:
        client_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        if reasoning_effort is not None:
            client_kwargs["reasoning_effort"] = reasoning_effort
    
    # 创建 ChatOpenAI 客户端（DeepSeek API 兼容 OpenAI 协议）
    # 注意：不自动添加回调处理器，让调用者自己控制输出方式
    client = ChatOpenAI(**client_kwargs)
    
    return client

# =============================================================================
# 创建用于 Metadata 元数据提取的 LLM 客户端
# =============================================================================
def get_llm_client(
    model: str = None,
    streaming: bool = False,
) -> ChatOpenAI:
    """
    获取用于元数据提取的 LLM 客户端。

    参数：
        model: 模型名称，如果为 None 则从 app.yaml 的 LLM_CONFIG 读取
        streaming: 是否启用流式输出，默认 False（元数据提取不需要流式）
    """
    from config.settings import LLM_CONFIG
    if model is None:
        model = LLM_CONFIG.get("model", "deepseek-v4-pro")
    timeout = LLM_CONFIG.get("timeout", 120)
    return create_deepseek_client(model=model, streaming=streaming, timeout=timeout)

# =============================================================================
# 带 LRU 缓存的 Embedding 包装器
# =============================================================================
class CachedEmbeddings:
    """
    带 LRU 缓存的 Embeddings 包装器，避免相同查询重复调用 embedding API。

    仅缓存 embed_query（运行时查询），不缓存 embed_documents（建索引时调用，每文档唯一）。
    线程安全，适用于并发检索场景——并行检索 battery+manual 时，同一 query 只计算一次 embedding。

    Attributes:
        _embeddings: 被包装的底层 Embeddings 实例
        _max_size: 缓存最大条目数
    """

    def __init__(self, embeddings, max_size: int = 256):
        """
        Args:
            embeddings: 底层 Embeddings 实例（如 DashScopeEmbeddings）
            max_size: 缓存最大条目数，超出后淘汰最久未使用的条目
        """
        self._embeddings = embeddings
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    def embed_query(self, text: str) -> List[float]:
        """
        对查询文本生成 embedding，带 LRU 缓存。

        同一 query（前 500 字符作为 key）只调用一次底层 API，
        后续命中缓存直接返回。线程安全：并行调用时第一个线程持锁计算，
        其余线程等锁后命中缓存。

        Args:
            text: 查询文本

        Returns:
            embedding 向量（List[float]）
        """
        cache_key = text[:500]

        with self._lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                logger.debug(f"Embedding cache hit for query: {text[:50]}...")
                return self._cache[cache_key]

        result = self._embeddings.embed_query(text)

        with self._lock:
            self._cache[cache_key] = result
            self._cache.move_to_end(cache_key)
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
            logger.debug(f"Embedding cache miss, cached query: {text[:50]}..., cache size: {len(self._cache)}")

        return result

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        对文档列表生成 embedding（不缓存，建索引时每个文档唯一）。

        Args:
            texts: 文档文本列表

        Returns:
            embedding 向量列表
        """
        return self._embeddings.embed_documents(texts)

    def __getattr__(self, name: str):
        """转发未定义的属性访问到底层 Embeddings 实例。"""
        return getattr(self._embeddings, name)


# =============================================================================
# 创建 Embedding 嵌入客户端
# =============================================================================
def get_embeddings(
    model: str = None,
) -> CachedEmbeddings:
    """
    获取 DashScope 嵌入模型客户端（带 LRU 缓存）。

    参数：
        model: 嵌入模型名称，如果为 None 则从 app.yaml 的 EMBEDDING_CONFIG 读取

    返回:
        CachedEmbeddings 实例，包装了 DashScopeEmbeddings
    """
    from config.settings import DASHSCOPE_API_KEY, EMBEDDING_CONFIG

    if model is None:
        model = EMBEDDING_CONFIG.get("model", "text-embedding-v4")

    raw_embeddings = DashScopeEmbeddings(
        model=model,
        dashscope_api_key=DASHSCOPE_API_KEY,
    )
    return CachedEmbeddings(raw_embeddings)
