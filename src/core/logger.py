"""日志模块：提供应用日志、Trace日志和Prompt日志的统一管理"""

import logging
import sys
from config.settings import LOG_LEVEL, LOG_FILE, LOGS_DIR


def setup_logger(name: str, log_file: str = None, level: int = None) -> logging.Logger:
    """
    创建并配置日志记录器。
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径，若为 None 则只输出到控制台
        level: 日志级别，默认使用 LOG_LEVEL 环境变量
    
    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    if level is None:
        level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
    logger.addHandler(ch)
    
    if log_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
        logger.addHandler(fh)
    
    return logger


logger = setup_logger("cw_ai", LOG_FILE)


def setup_trace_logger() -> logging.Logger:
    """创建 Trace 专用日志记录器，写入 logs/trace.log"""
    trace_logger = logging.getLogger("cw_ai.trace")
    if trace_logger.handlers:
        return trace_logger
    
    trace_logger.setLevel(logging.DEBUG)
    trace_logger.propagate = False
    
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    trace_file = LOGS_DIR / "trace.log"
    fh = logging.FileHandler(trace_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [TRACE] %(message)s"))
    trace_logger.addHandler(fh)
    
    return trace_logger


def setup_prompt_logger() -> logging.Logger:
    """创建 Prompt 专用日志记录器，写入 logs/prompt.log"""
    prompt_logger = logging.getLogger("cw_ai.prompt")
    if prompt_logger.handlers:
        return prompt_logger
    
    prompt_logger.setLevel(logging.DEBUG)
    prompt_logger.propagate = False
    
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    prompt_file = LOGS_DIR / "prompt.log"
    fh = logging.FileHandler(prompt_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [PROMPT] %(message)s"))
    prompt_logger.addHandler(fh)
    
    return prompt_logger


trace_logger = setup_trace_logger()
prompt_logger = setup_prompt_logger()
