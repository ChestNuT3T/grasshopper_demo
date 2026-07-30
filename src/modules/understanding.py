"""理解模块：提供用户意图理解、关键词提取和检索目标确定功能"""

import os
import time
from typing import Optional, List

from pydantic import BaseModel, Field

from src.core.client import create_deepseek_client
from src.core.tools import parse_json_with_fallback, safe_print
from src.modules.prompts import UNIFIED_PROMPT_CONFIG, MERGE_PROMPT_CONFIG
from src.core.logger import logger


class Keywords(BaseModel):
    component: str = Field(default="无", description="构件（立柱、横梁、玻璃面板、型材、埋件、转接件等）")
    geometry_type: str = Field(default="无", description="几何类型（Brep、Surface、Curve、Plane、Mesh、Point、Polyline 等）")
    battery_plugin: str = Field(default="无", description="电池/插件（原生电池如 Isotrim、Divide Surface，插件如 Elefront、Human）")
    operation: str = Field(default="无", description="操作（分组、排序、编号、属性写入、烘焙、偏移、挤出等）")
    data_type: str = Field(default="无", description="数据类型（树、列表、分支、路径、法线、平面等）")
    issue_phenomenon: str = Field(default="无", description="问题现象（报错、数据为空、变形、重叠、超出范围等）")
    curtain_wall_type: str = Field(default="未提及", description="幕墙类型（常规、单曲、双曲）")


class UnderstandingResult(BaseModel):
    enriched_question: str = Field(description="丰富后的标准化问题描述")
    task_type: str = Field(description="任务类型：故障诊断/知识查询/搭建指导/数据处理")
    keywords: Keywords = Field(default_factory=Keywords)
    user_input: str = Field(default="", description="原始用户输入")
    retrieval_target: str = Field(default="both", description="检索目标：battery_only/manual_only/both")
    retrieval_target_reason: str = Field(default="", description="检索目标判断理由")
    needs_clarification: bool = Field(default=False, description="是否需要追问用户补充信息")
    clarification_message: str = Field(default="", description="追问提示信息")


def unified_understand(user_input: str, chat_history: List = None) -> UnderstandingResult:
    if chat_history is None:
        chat_history = []

    start_time = time.time()
    logger.info(f"unified_understand started - input: {user_input[:100]}...")

    client = create_deepseek_client(model="deepseek-v4-pro", streaming=False)

    try:
        system_content = UNIFIED_PROMPT_CONFIG.get("system", "").format(
            user_input=user_input,
            chat_history=chat_history
        )
        user_template = UNIFIED_PROMPT_CONFIG.get("user_template", "{user_input}")
        user_content = user_template.format(user_input=user_input)
        
        messages = [
            {"role": "system", "content": system_content},
            *chat_history,
            {"role": "user", "content": user_content}
        ]
        result = client.invoke(messages)
        result_text = getattr(result, 'content', str(result))
        parsed_result = parse_json_with_fallback(result_text)

        elapsed = time.time() - start_time

        if isinstance(parsed_result, dict):
            if "keyword" in parsed_result and "keywords" not in parsed_result:
                parsed_result["keywords"] = parsed_result.pop("keyword")
            if "reconstructed_question" in parsed_result and "enriched_question" not in parsed_result:
                parsed_result["enriched_question"] = parsed_result.pop("reconstructed_question")
            if not parsed_result.get("user_input"):
                parsed_result["user_input"] = user_input

            result_obj = UnderstandingResult(**parsed_result)
            
            completeness = check_completeness(result_obj)
            result_obj.needs_clarification = not completeness["is_complete"]
            result_obj.clarification_message = completeness["message"]
            
            logger.info(f"unified_understand completed - task_type: {result_obj.task_type}, elapsed: {elapsed:.2f}s")
            return result_obj

        raise ValueError("解析结果不是字典")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"unified_understand failed - elapsed: {elapsed:.2f}s, error: {e}")
        return UnderstandingResult(
            enriched_question="请检查您的问题是否和Rhino及Grasshopper建模有关。",
            task_type="其他",
            keywords=Keywords(),
            user_input=user_input,
            retrieval_target="both",
            retrieval_target_reason="解析失败，默认使用both",
            needs_clarification=False,
            clarification_message=""
        )


def check_completeness(result: UnderstandingResult) -> dict:
    """
    检查理解结果是否完整，根据任务类型判断是否缺少必要信息。
    
    Args:
        result: UnderstandingResult 对象，包含任务类型和关键词
        
    Returns:
        字典，包含 is_complete（是否完整）和 message（提示信息）
    """
    missing_fields = []
    task_type = result.task_type

    if task_type == "搭建指导":
        if result.keywords.operation == "无":
            missing_fields.append("操作类型")
        if result.keywords.geometry_type == "无":
            missing_fields.append("几何类型")
    elif task_type == "故障诊断":
        if result.keywords.issue_phenomenon == "无":
            missing_fields.append("问题现象")
    elif task_type == "数据处理":
        if result.keywords.operation == "无":
            missing_fields.append("数据操作类型")
        if result.keywords.data_type == "无":
            missing_fields.append("数据类型")

    if missing_fields:
        return {
            "is_complete": False,
            "message": f"信息不完整，缺少：{', '.join(missing_fields)}。请补充相关信息。"
        }

    return {
        "is_complete": True,
        "message": "信息完整，可以继续分析。"
    }


def lightweight_merge(original: UnderstandingResult, supplement: str) -> UnderstandingResult:
    """
    将用户补充的信息与原始理解结果合并，生成更新后的理解结果。
    
    Args:
        original: 原始的 UnderstandingResult 对象
        supplement: 用户补充的信息字符串
        
    Returns:
        更新后的 UnderstandingResult 对象，若合并失败则返回原始结果
    """
    client = create_deepseek_client(model="deepseek-v4-pro", streaming=False)

    try:
        system_content = MERGE_PROMPT_CONFIG.get("system", "").format(
            original_result=original.model_dump_json(),
            supplement=supplement
        )
        user_template = MERGE_PROMPT_CONFIG.get("user_template", "{supplement}")
        user_content = user_template.format(
            original_result=original.model_dump_json(),
            supplement=supplement
        )
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        result = client.invoke(messages)
        result_text = getattr(result, 'content', str(result))
        parsed_result = parse_json_with_fallback(result_text)

        if isinstance(parsed_result, dict):
            if "keyword" in parsed_result and "keywords" not in parsed_result:
                parsed_result["keywords"] = parsed_result.pop("keyword")
            if "reconstructed_question" in parsed_result and "enriched_question" not in parsed_result:
                parsed_result["enriched_question"] = parsed_result.pop("reconstructed_question")
            if not parsed_result.get("user_input"):
                parsed_result["user_input"] = original.user_input

            return UnderstandingResult(**parsed_result)

        raise ValueError("解析结果不是字典")
    except Exception as e:
        logger.error(f"lightweight_merge failed: {e}")
        return original


def interactive_completion(user_input: str, chat_history: List = None, max_retries: int = 3) -> UnderstandingResult:
    """
    交互式补全：如果理解结果不完整，通过追问用户获取缺失信息。
    注意：Web环境下不会阻塞等待输入，而是返回 needs_clarification=True 的结果，
    由前端处理追问逻辑。
    
    Args:
        user_input: 用户原始输入
        chat_history: 聊天历史列表
        max_retries: 最大追问次数，默认为3次
        
    Returns:
        UnderstandingResult 对象，needs_clarification 字段标记是否需要前端追问
    """
    result = unified_understand(user_input, chat_history)
    
    if result.needs_clarification:
        logger.info(f"理解结果不完整，需要追问：{result.clarification_message}")
        safe_print(f"追问：{result.clarification_message}")
        return result

    return result


def determine_retrieval_target(user_input: str, keywords: dict) -> str:
    """
    【已废弃】此函数已被 LLM 判断替代，仅作为兜底保留。
    
    原功能：根据用户输入和提取的关键词判断检索目标。
    
    Args:
        user_input: Original user input.
        keywords: Keywords dictionary from unified_understand.
    
    Returns:
        Retrieval target: "battery_only", "manual_only", or "both".
        
    Note:
        建议使用 unified_understand 返回的 UnderstandingResult.retrieval_target 字段，
        该字段由 LLM 直接判断，语义理解更准确。
    """
    battery_plugin = keywords.get("battery_plugin", "")
    operation = keywords.get("operation", "")
    
    if battery_plugin and battery_plugin != "无":
        logger.info(f"Retrieval target (fallback): battery_only (battery_plugin={battery_plugin})")
        return "battery_only"
    
    manual_operations = {"设置", "选项", "显示", "渲染", "视图", "建模"}
    if operation in manual_operations:
        logger.info(f"Retrieval target (fallback): manual_only (operation={operation})")
        return "manual_only"
    
    battery_signal = ["电池", "电池组", "运算器", "Grasshopper", "GH", "插件", "面板", "菜单路径"]
    manual_signal = ["Rhino", "犀牛", "命令", "设置", "选项", "视图", "建模", "显示模式", "图层", "材质", "渲染"]
    
    battery_score = sum(1 for word in battery_signal if word in user_input)
    manual_score = sum(1 for word in manual_signal if word in user_input)
    
    if battery_score > manual_score:
        logger.info(f"Retrieval target (fallback): battery_only (battery_score={battery_score}, manual_score={manual_score})")
        return "battery_only"
    elif manual_score > battery_score:
        logger.info(f"Retrieval target (fallback): manual_only (battery_score={battery_score}, manual_score={manual_score})")
        return "manual_only"
    else:
        logger.info(f"Retrieval target (fallback): both (battery_score={battery_score}, manual_score={manual_score})")
        return "both"


def should_use_interactive() -> bool:
    """
    判断是否启用交互式补全模式。
    
    Returns:
        是否启用交互式补全，从 settings.INTERACTIVE_ENABLED 读取
    """
    from config.settings import INTERACTIVE_ENABLED
    return INTERACTIVE_ENABLED