"""理解模块：提供用户意图理解、关键词提取和检索目标确定功能。

本模块通过 LLM 完成意图分类、关键词提取、检索目标判断和问题丰富。
关键设计：
1. 启用 DeepSeek JSON Output 模式（response_format=json_object），强制 LLM 输出合法 JSON
2. 对临时性错误（网络超时、空 content）自动重试，最多 2 次
3. 系统级错误（API 不可用）上抛给调用方，不静默返回空壳
4. 输入级错误（与 Rhino/GH 无关）返回明确的"其他"分类提示
"""

import os
import time
from typing import Optional, List

from pydantic import BaseModel, Field, ValidationError

from src.core.client import create_deepseek_client
from src.core.tools import parse_json_with_fallback, safe_print
from src.modules.prompts import UNIFIED_PROMPT_CONFIG, MERGE_PROMPT_CONFIG
from src.core.logger import logger


# =============================================================================
# 自定义异常：区分系统错误和输入错误
# =============================================================================
class LLMServiceError(Exception):
    """LLM 服务不可用错误（网络超时、API 限流、空 content 等），应上抛或重试。"""
    pass


class LLMOutputError(Exception):
    """LLM 输出解析失败错误（JSON 格式错误、字段缺失等），可重试。"""
    pass


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


def _invoke_llm_with_retry(
    client,
    messages: list,
    user_input: str,
    max_retries: int = 2
) -> dict:
    """
    带 retry 的 LLM 调用，区分临时性错误（重试）和致命错误（上抛）。

    临时性错误：网络超时、API 限流、空 content（DeepSeek JSON mode 已知问题）
    致命错误：Pydantic 校验失败（字段缺失/类型错误，重试也没用）

    Args:
        client: ChatOpenAI 客户端实例
        messages: 消息列表
        user_input: 原始用户输入（用于兜底填充 user_input 字段）
        max_retries: 最大重试次数，默认 2 次

    Returns:
        解析后的字典

    Raises:
        LLMServiceError: 重试后仍失败（网络/API 不可用）
        LLMOutputError: LLM 输出无法解析（JSON 格式错误、字段缺失）
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            result = client.invoke(messages)
            result_text = getattr(result, 'content', str(result))

            # 应对 DeepSeek JSON mode 已知问题：有概率返回空 content
            if not result_text or not result_text.strip():
                logger.warning(f"LLM returned empty content (attempt {attempt + 1}/{max_retries + 1})")
                last_error = LLMServiceError("LLM returned empty content")
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                raise last_error

            parsed_result = parse_json_with_fallback(result_text)

            if not isinstance(parsed_result, dict) or not parsed_result:
                raise LLMOutputError(f"LLM output is not a valid dict: {result_text[:100]}...")

            # 兜底填充 user_input（不再做 keyword/keywords 字段名 hack，JSON mode 应保证字段名稳定）
            if not parsed_result.get("user_input"):
                parsed_result["user_input"] = user_input

            return parsed_result

        except (LLMServiceError, LLMOutputError):
            raise
        except ValidationError as e:
            raise LLMOutputError(f"Pydantic validation failed: {e}")
        except Exception as e:
            # 网络超时、API 限流等临时性错误，重试
            error_msg = str(e).lower()
            is_transient = any(kw in error_msg for kw in [
                "timeout", "timed out", "rate limit", "429", "503",
                "connection", "apitimeout", "service unavailable"
            ])
            last_error = e
            if is_transient and attempt < max_retries:
                logger.warning(f"LLM transient error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                time.sleep(2)
                continue
            raise LLMServiceError(f"LLM service error: {e}") from e

    raise LLMServiceError(f"LLM call failed after {max_retries + 1} attempts: {last_error}")


def unified_understand(user_input: str, chat_history: List = None) -> UnderstandingResult:
    """
    用户意图理解：一次 LLM 调用完成纠错、补全、关键词提取、意图分类、检索目标判断。

    改造点：
    1. 启用 json_mode（response_format=json_object），强制 LLM 输出合法 JSON
    2. 临时性错误（网络/限流/空 content）自动重试 2 次
    3. 系统级错误（API 不可用）上抛 LLMServiceError，不返回空壳
    4. 移除 keyword→keywords、reconstructed_question→enriched_question 的字段名 hack

    Args:
        user_input: 用户原始输入
        chat_history: 聊天历史列表

    Returns:
        UnderstandingResult 对象

    Raises:
        LLMServiceError: LLM 服务不可用（重试后仍失败）
        LLMOutputError: LLM 输出解析失败
    """
    if chat_history is None:
        chat_history = []

    start_time = time.time()
    logger.info(f"unified_understand started - input: {user_input[:100]}...")

    # 启用 JSON mode，streaming=False（理解阶段不需要流式）
    client = create_deepseek_client(
        model="deepseek-v4-pro",
        streaming=False,
        json_mode=True,
        max_tokens=2000
    )

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

        parsed_result = _invoke_llm_with_retry(client, messages, user_input)

        elapsed = time.time() - start_time
        result_obj = UnderstandingResult(**parsed_result)

        completeness = check_completeness(result_obj)
        result_obj.needs_clarification = not completeness["is_complete"]
        result_obj.clarification_message = completeness["message"]

        logger.info(f"unified_understand completed - task_type: {result_obj.task_type}, elapsed: {elapsed:.2f}s")
        return result_obj

    except (LLMServiceError, LLMOutputError):
        # 系统级错误上抛，不返回空壳
        elapsed = time.time() - start_time
        logger.error(f"unified_understand failed (system error) - elapsed: {elapsed:.2f}s")
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"unified_understand failed (unexpected) - elapsed: {elapsed:.2f}s, error: {e}", exc_info=True)
        raise LLMServiceError(f"Unexpected error in unified_understand: {e}") from e


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

    改造点：
    1. 启用 json_mode 强制 JSON 输出
    2. 移除字段名 hack
    3. 失败时在 original 上打 merge_failed 标记（不再静默返回，让用户知道合并失败）

    Args:
        original: 原始的 UnderstandingResult 对象
        supplement: 用户补充的信息字符串

    Returns:
        更新后的 UnderstandingResult 对象。合并失败时返回原对象，但 enriched_question
        前会加"[补充信息合并失败]"提示，让用户感知到补充信息未生效。

    Raises:
        LLMServiceError: LLM 服务不可用（重试后仍失败）
    """
    # 启用 JSON mode
    client = create_deepseek_client(
        model="deepseek-v4-pro",
        streaming=False,
        json_mode=True,
        max_tokens=2000
    )

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

        parsed_result = _invoke_llm_with_retry(client, messages, original.user_input)

        if not parsed_result.get("user_input"):
            parsed_result["user_input"] = original.user_input

        return UnderstandingResult(**parsed_result)

    except (LLMServiceError, LLMOutputError) as e:
        # 合并失败时显式提示用户，不再静默返回原值
        logger.error(f"lightweight_merge failed: {e}")
        merged = original.model_copy(deep=True)
        merged.enriched_question = f"[补充信息合并失败，请重新描述] {original.enriched_question}"
        return merged
    except Exception as e:
        logger.error(f"lightweight_merge failed (unexpected): {e}", exc_info=True)
        merged = original.model_copy(deep=True)
        merged.enriched_question = f"[补充信息合并失败，请重新描述] {original.enriched_question}"
        return merged


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


def should_use_interactive() -> bool:
    """
    判断是否启用交互式补全模式。
    
    Returns:
        是否启用交互式补全，从 settings.INTERACTIVE_ENABLED 读取
    """
    from config.settings import INTERACTIVE_ENABLED
    return INTERACTIVE_ENABLED