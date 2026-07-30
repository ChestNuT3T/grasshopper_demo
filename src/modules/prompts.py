"""提示词模块：提供提示词模板的加载、格式化和管理功能"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from config.settings import PROJECT_ROOT

_prompt_cache: Optional[Dict[str, Any]] = None


def load_prompt_config() -> Dict[str, Any]:
    """
    加载提示词配置（带缓存）。
    
    从 configs/prompts.yaml 加载所有提示词模板。若文件不存在，抛出 FileNotFoundError。
    
    Returns:
        提示词配置字典，键为模板名称，值为 {"system": "...", "user_template": "..."}
        
    Raises:
        FileNotFoundError: 当 prompts.yaml 不存在时抛出
    """
    global _prompt_cache
    if _prompt_cache is not None:
        return _prompt_cache

    prompt_path = PROJECT_ROOT / "configs" / "prompts.yaml"
    if not prompt_path.exists():
        raise FileNotFoundError(f"提示词配置文件不存在: {prompt_path}")

    with open(prompt_path, 'r', encoding='utf-8') as f:
        _prompt_cache = yaml.safe_load(f) or {}
    return _prompt_cache


def get_prompt_template(name: str) -> Dict[str, str]:
    """
    获取指定名称的提示词模板。
    
    Args:
        name: 模板名称
        
    Returns:
        模板字典 {"system": "...", "user_template": "..."}，若模板不存在返回空字典
    """
    config = load_prompt_config()
    return config.get(name, {})


def format_prompt(name: str, **kwargs) -> Dict[str, str]:
    """
    获取并格式化提示词。
    
    Args:
        name: 模板名称
        **kwargs: 用于格式化模板的参数
        
    Returns:
        格式化后的提示词 {"system": "...", "user": "..."}
        
    Raises:
        KeyError: 当模板不存在时抛出
    """
    template = get_prompt_template(name)
    if not template:
        raise KeyError(f"提示词模板 '{name}' 不存在")
    
    system = template.get("system", "").format(**kwargs) if template.get("system") else ""
    user = template.get("user_template", "{user_input}").format(**kwargs) if template.get("user_template") else ""
    return {"system": system, "user": user}


def get_unified_prompt() -> Dict[str, str]:
    """
    获取统一理解提示词。
    
    Returns:
        格式化后的提示词 {"system": "...", "user": "..."}
    """
    return get_prompt_template("unified_understanding")


def get_merge_prompt() -> Dict[str, str]:
    """
    获取合并理解提示词。
    
    Returns:
        格式化后的提示词 {"system": "...", "user": "..."}
    """
    return get_prompt_template("merge")


UNIFIED_PROMPT_CONFIG = get_unified_prompt()
MERGE_PROMPT_CONFIG = get_merge_prompt()
