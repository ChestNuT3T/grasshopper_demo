"""Trace模块：定义检索和链路上的追踪数据结构，用于记录和分析系统运行过程"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import time


@dataclass(slots=True)
class RetrievalTrace:
    """检索追踪数据结构，记录单次检索的完整信息"""
    
    query: str                      # 原始查询词
    rewritten_query: str = ""       # 重写后的查询词
    match_mode: str = "unknown"     # 匹配模式："exact"（精确匹配）/ "vector"（向量检索）/ "hybrid"（混合检索）
    metadata_filter: Dict[str, Any] = field(default_factory=dict)  # 元数据过滤条件
    catalog_match: Optional[Dict] = None  # 目录精确匹配结果
    stage_counts: Dict[str, int] = field(default_factory=dict)     # 各阶段检索数量（battery/manual）
    stage_times: Dict[str, float] = field(default_factory=dict)    # 各阶段耗时（prefilter_ms/vector_ms/rerank_ms/total_ms）
    fallback_used: bool = False     # 是否使用了降级策略
    total_time_ms: float = 0.0      # 总耗时（毫秒）
    results: List[Dict] = field(default_factory=list)              # 检索结果列表，包含 chunk_id、section、score 等
    errors: List[str] = field(default_factory=list)                # 错误信息列表

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "rewritten_query": self.rewritten_query,
            "match_mode": self.match_mode,
            "metadata_filter": self.metadata_filter,
            "catalog_match": self.catalog_match,
            "stage_counts": self.stage_counts,
            "stage_times": {k: round(v, 2) for k, v in self.stage_times.items()},
            "fallback_used": self.fallback_used,
            "total_time_ms": round(self.total_time_ms, 2),
            "results": self.results[:5],
            "errors": self.errors
        }


@dataclass(slots=True)
class ChainTrace:
    """完整链路追踪数据结构，记录单次对话的完整处理过程"""
    
    session_id: str                          # 会话ID
    user_input: str                          # 用户原始输入
    task_type: str = "其他"                  # 任务类型：故障诊断/知识查询/搭建指导/数据处理
    keywords: Dict[str, str] = field(default_factory=dict)  # 提取的关键词
    retrieval: Optional[RetrievalTrace] = None  # 检索追踪信息
    prompt: str = ""                         # 发送给LLM的提示词
    response: str = ""                       # LLM生成的回答
    reasoning: Optional[str] = None          # LLM的推理过程
    total_time_ms: float = 0.0               # 总耗时（毫秒）
    token_usage: Dict[str, int] = field(default_factory=dict)  # Token使用情况
    start_time: float = field(default_factory=time.time)       # 开始时间戳

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_input": self.user_input,
            "task_type": self.task_type,
            "keywords": self.keywords,
            "retrieval": self.retrieval.to_dict() if self.retrieval else None,
            "prompt": self.prompt,
            "response": self.response[:500] + "..." if len(self.response) > 500 else self.response,
            "reasoning": self.reasoning,
            "total_time_ms": round(self.total_time_ms, 2),
            "token_usage": self.token_usage
        }