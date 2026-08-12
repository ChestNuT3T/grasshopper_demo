"""
配置加载优先级：
1. 环境变量（.env）—— 最高优先级，用于 API Keys 和运行时覆盖
2. YAML 配置（app.yaml）—— 主要配置源，包含 RAG、Embedding、Rerank 等
3. 代码默认值——最低优先级，仅当 YAML 和环境变量均未设置时使用
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAG_DIR = DATA_DIR / "RAG"
CHROMA_DIR = DATA_DIR / "chroma_db"
LOGS_DIR = PROJECT_ROOT / "logs"
CHAT_HISTORY_DIR = PROJECT_ROOT / "chat_history"
CHAT_DB_PATH = CHAT_HISTORY_DIR / "chat.db"
CONFIG_DIR = PROJECT_ROOT / "configs"

PROCESSED_DIR = DATA_DIR / "processed"
BATTERY_PROCESSED_DIR = PROCESSED_DIR / "battery"
MANUAL_PROCESSED_DIR = PROCESSED_DIR / "manual"

BATTERY_LEDGER_PATH = BATTERY_PROCESSED_DIR / "grasshopper_chunks.jsonl"
BATTERY_CATALOG_PATH = BATTERY_PROCESSED_DIR / "grasshopper_catalog.json"
MANUAL_LEDGER_PATH = MANUAL_PROCESSED_DIR / "rhino8_chunks.jsonl"
MANUAL_CATALOG_PATH = MANUAL_PROCESSED_DIR / "rhino8_catalog.json"

JSON_PATH = str(RAG_DIR / "crawl_results_filtered.json")
PDF_PATH = str(RAG_DIR / "Rhino User's Guide for Windows.pdf")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-pro")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = str(LOGS_DIR / "app.log")

_config = None


def load_yaml_config() -> dict:
    """Load YAML configuration from configs/app.yaml"""
    global _config
    if _config is None:
        config_path = CONFIG_DIR / "app.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                _config = yaml.safe_load(f)
        else:
            _config = {}
    return _config


def get_config() -> dict:
    """Get the loaded YAML configuration"""
    return load_yaml_config()


if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY 未设置")

_config = load_yaml_config()

RAG_CONFIG = _config.get("rag", {})
EMBEDDING_CONFIG = _config.get("embedding", {})
RERANK_CONFIG = _config.get("rerank", {})
PATHS_CONFIG = _config.get("paths", {})
RETRIEVAL_CONFIG = _config.get("retrieval", {})


def get_retrieval_config(task_type: str = "") -> dict:
    """
    获取检索配置，支持按任务类型覆盖默认值。
    
    Args:
        task_type: 任务类型（如 "knowledge_query", "fault_diagnosis"）
        
    Returns:
        检索配置字典，包含 battery_top_k, manual_top_k, rerank_top_k 等
    """
    config = {
        "battery_top_k": RETRIEVAL_CONFIG.get("battery_top_k", 10),
        "manual_top_k": RETRIEVAL_CONFIG.get("manual_top_k", 10),
        "rerank_top_k": RETRIEVAL_CONFIG.get("rerank_top_k", 5),
        "rerank_cache_max_size": RETRIEVAL_CONFIG.get("rerank_cache_max_size", 128),
        "fallback_min_docs": RETRIEVAL_CONFIG.get("fallback_min_docs", 3),
        "candidate_multiplier": RETRIEVAL_CONFIG.get("candidate_multiplier", 2),
    }
    
    overrides = RETRIEVAL_CONFIG.get("task_type_overrides", {})
    if task_type and task_type in overrides:
        config.update(overrides[task_type])
    
    return config


def get_default_retriever_k() -> int:
    """
    获取检索器初始化时的默认召回数（从配置读取，用于加载检索器）。
    
    Returns:
        默认召回数，优先使用 battery_top_k，默认值为 10
    """
    return RETRIEVAL_CONFIG.get("battery_top_k", 10)

LLM_CONFIG = {
    "model": os.getenv("LLM_MODEL", _config.get("llm", {}).get("model", "deepseek-v4-pro")),
    "timeout": int(os.getenv("LLM_TIMEOUT", str(_config.get("llm", {}).get("timeout", 120)))),
}

RAG_ENABLED = os.getenv("ENABLE_RAG", str(RAG_CONFIG.get("enabled", True))).lower() == "true"
INTERACTIVE_ENABLED = os.getenv("ENABLE_INTERACTIVE_COMPLETION", str(RAG_CONFIG.get("interactive", False))).lower() == "true"

if not DASHSCOPE_API_KEY:
    import warnings
    warnings.warn("DASHSCOPE_API_KEY 未设置，重排序功能将不可用", UserWarning)