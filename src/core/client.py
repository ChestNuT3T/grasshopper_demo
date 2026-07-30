"""
=============================================================================
        客户端模块：提供 LLM 客户端和 Embedding 客户端的创建和管理功能
=============================================================================
"""

import os
import time
from typing import Dict, Optional, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.embeddings.dashscope import DashScopeEmbeddings

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
    reasoning_effort: str = "high"
) -> ChatOpenAI:
    from config.settings import DEEPSEEK_API_KEY
    api_key = DEEPSEEK_API_KEY
    
    # 检查 API Key 是否存在
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 环境变量未设置")
    
    # 构建 model_kwargs，仅在流式模式下设置 stream_options
    model_kwargs = {}
    if streaming:
        model_kwargs["stream_options"] = {"include_usage": True}
    
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
# 创建 Embedding 嵌入客户端
# =============================================================================
def get_embeddings(
    model: str = None,
) -> DashScopeEmbeddings:
    """
    获取 DashScope 嵌入模型客户端。

    参数：
        model: 嵌入模型名称，如果为 None 则从 app.yaml 的 EMBEDDING_CONFIG 读取
    """
    from config.settings import DASHSCOPE_API_KEY, EMBEDDING_CONFIG
    
    if model is None:
        model = EMBEDDING_CONFIG.get("model", "text-embedding-v4")
    
    return DashScopeEmbeddings(
        model=model,
        dashscope_api_key=DASHSCOPE_API_KEY,
    )
