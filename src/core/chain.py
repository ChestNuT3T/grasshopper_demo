"""
=============================================================================
        核心链模块：理解、检索、生成
=============================================================================
"""

import os
import time
import json
from dotenv import load_dotenv
from typing import Dict, Any, List

from config.settings import RAG_ENABLED

from langchain_core.globals import set_debug
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

from src.core.client import create_deepseek_client
from openai import OpenAI
from src.core.tools import safe_print, get_session_history, get_chat_history
from src.modules.prompts import format_prompt
from src.modules.understanding import unified_understand, interactive_completion, should_use_interactive
from src.modules.retrieval import retrieve_reference_docs, build_battery_retriever, load_battery_retriever, load_manual_retriever
from src.core.logger import logger, trace_logger, prompt_logger
from src.core.trace import ChainTrace

load_dotenv()
battery_retriever = None
manual_retriever = None

# =============================================================================
# 获取会话历史（从 config 中提取 session_id，获取该会话的历史消息列表）
# =============================================================================
def get_history_for_preprocessing(input_data, config=None):
    """
    步骤1：获取历史消息用于预处理阶段
    
    参数:
        input_data: 输入数据
        config: 配置信息，包含 session_id
    
    返回:
        包含 chat_history 的字典（chat_history 为 BaseMessage 列表）
    """
    result = dict(input_data)

    # 从 config 中提取 session_id
    session_id = None
    if config and isinstance(config, dict):
        session_id = config.get("configurable", {}).get("session_id")
    # 如果 session_id 存在，获取历史消息
    if session_id:
        history_messages = get_chat_history(session_id, format_output=False)
        result["chat_history"] = history_messages if history_messages else []
    else:
        result["chat_history"] = []
    
    return result

# =============================================================================
# 统一理解（task_type 分类 + keyword 提取 + 问题 Enrich）
# =============================================================================
def unified_understand_wrapper(input_data: dict) -> dict:
    """
    步骤2：统一理解（一次LLM调用完成关键词提取、意图分类和问题丰富）
    
    调用 unified_understand 或 interactive_completion（根据环境变量控制），
    将 UnderstandingResult 转换为下游兼容格式，并确定检索目标。
    
    参数:
        input_data: 字典，包含 user_input 和 chat_history
    
    返回:
        兼容原有接口的字典格式，包含 extract_result、classify_result 和 retrieval_target
    """
    user_input = input_data.get("user_input", "")
    chat_history = input_data.get("chat_history", [])

    # 根据 interactive 配置决定是否使用 interactive_completion
    if should_use_interactive():
        result = interactive_completion(user_input, chat_history)
    else:
        result = unified_understand(user_input, chat_history)

    keywords_dict = result.keywords.model_dump()

    # 检索目标决定从哪个知识库检索
    retrieval_target = result.retrieval_target
    retrieval_target_reason = result.retrieval_target_reason

    return {
        "extract_result": {
            "keyword": keywords_dict,
            "reconstructed_question": result.enriched_question,
            "user_input": result.user_input,
            "success": True,
            "error_message": ""
        },
        "classify_result": {
            "task_type": result.task_type,
            "success": True,
            "error_message": ""
        },
        "user_input": result.user_input,
        "original_query": result.user_input,
        "retrieval_target": retrieval_target,
        "retrieval_target_reason": retrieval_target_reason,
        "needs_clarification": getattr(result, "needs_clarification", False),
        "clarification_message": getattr(result, "clarification_message", "")
    }

# =============================================================================
# 构建Prompt（根据 unified_understand_wrapper 函数输出结果，拼装Prompt）
# =============================================================================
def build_analys_prompt(data: dict) -> list:
    """
    步骤3：构建分析阶段的消息列表
    
    根据预处理结果（关键词、任务类型、重构问题等）构建完整的提示词消息列表，
    供 analys_result 调用 LLM 使用。
    
    参数:
        data: 包含 extract_result, classify_result, original_query 的字典
    
    返回:
        填充好的消息列表（BaseMessage列表，不含历史消息）
    """ 
    extract_result = data.get("extract_result", {})
    classify_result = data.get("classify_result", {})
    keyword = extract_result.get("keyword", {})

    reference_docs = data.get("reference_docs", "")
    component = keyword.get("component", "无")
    geometry_type = keyword.get("geometry_type", "无")
    battery_plugin = keyword.get("battery_plugin", "无")
    operation = keyword.get("operation", "无")
    data_type = keyword.get("data_type", "无")
    issue_phenomenon = keyword.get("issue_phenomenon", "无")
    curtain_wall_type = keyword.get("curtain_wall_type", "未提及")
    reconstructed_question = extract_result.get("reconstructed_question", "")
    task_type = classify_result.get("task_type", "其他")

    template_name_map = {
        "故障诊断": "diagnosis",
        "知识查询": "knowledge",
        "数据处理": "data_processing",
        "搭建指导": "building_guide",
    }
    template_name = template_name_map.get(task_type, "analysis")

    # 处理历史消息和当前用户输入
    user_input_data = data.get("user_input", "")

    if isinstance(user_input_data, list) and user_input_data:
        current_message = user_input_data[-1]
        current_user_text = current_message.content if hasattr(current_message, 'content') else str(current_message)
        history = user_input_data[:-1]
    else:
        current_user_text = str(user_input_data)
        history = []

    prompt_dict = format_prompt(
        template_name,
        user_input=current_user_text,
        chat_history=history,
        reference_docs=reference_docs,
        component=component,
        geometry_type=geometry_type,
        battery_plugin=battery_plugin,
        operation=operation,
        data_type=data_type,
        issue_phenomenon=issue_phenomenon,
        curtain_wall_type=curtain_wall_type,
        reconstructed_question=reconstructed_question,
        task_type=task_type,
    )

    messages = [
        SystemMessage(content=prompt_dict["system"]),
        *history,
        HumanMessage(content=prompt_dict["user"]),
    ]
    return messages

# =============================================================================
# 调用 LLM 流式生成（创建 DeepSeek 客户端，以流式方式调用模型，实时输出内容）
# =============================================================================
def analys_result(messages: list, session_id: str = "", task_type: str = "", keywords: dict = None) -> str:
    """
    步骤4：调用 DeepSeek 大模型进行分析（流式输出）
    
    参数:
        messages: 消息列表（BaseMessage列表）
        session_id: 会话ID（用于追踪）
        task_type: 任务类型（如"故障诊断"、"知识查询"）
        keywords: 提取的关键词字典
    
    返回:
        Markdown 格式的详细解决方案（字符串）
    """
    if keywords is None:
        keywords = {}
    total_start_time = time.time()
    logger.info(f"analys_result called with {len(messages)} messages")

    # 提取 system 和 user 内容用于日志
    system_content = ""
    user_content = ""
    for msg in messages:
        if hasattr(msg, 'type') and msg.type == 'system':
            system_content = getattr(msg, 'content', '')
        elif hasattr(msg, 'type') and msg.type == 'human':
            user_content = getattr(msg, 'content', '')
    
    logger.debug(f"analys_result system prompt: {system_content[:200]}...")
    logger.info(f"analys_result - {len(messages)} messages, system prompt: {system_content[:100]}...")
    
    full_prompt = f"System: {system_content}\n\nUser: {user_content}"

    # 根据 LOG_LEVEL 决定记录完整 Prompt 还是仅记录摘要
    # LOG_LEVEL=DEBUG 时记录完整内容，否则只记录摘要信息
    from config.settings import LOG_LEVEL
    if LOG_LEVEL.upper() == "DEBUG":
        messages_serialized = []
        for msg in messages:
            msg_dict = {
                "type": getattr(msg, 'type', ''),
                "content": getattr(msg, 'content', '')
            }
            messages_serialized.append(msg_dict)
        
        prompt_logger.info(json.dumps({
            "prompt_length": len(full_prompt),
            "message_count": len(messages),
            "system_length": len(system_content),
            "user_length": len(user_content),
            "full_messages": messages_serialized,
            "full_system_prompt": system_content,
            "full_user_prompt": user_content
        }, ensure_ascii=False))
    else:
        prompt_logger.info(json.dumps({
            "prompt_length": len(full_prompt),
            "message_count": len(messages),
            "system_length": len(system_content),
            "user_length": len(user_content)
        }, ensure_ascii=False))

    # 创建 DeepSeek 客户端
    client = create_deepseek_client(streaming=True, timeout=300)

    try:
        logger.info("Calling DeepSeek model with streaming...")

        result = ""
        reasoning_content = ""
        chunk_count = 0
        token_count = 0
        first_token_time = None
        
        token_usage = {}

        # stream 输出
        for chunk in client.stream(messages):
            chunk_count += 1
            if hasattr(chunk, 'content') and chunk.content:
                if not first_token_time:
                    first_token_time = time.time()
                token_count += 1
                safe_print(chunk.content)
                result += chunk.content
            
            if hasattr(chunk, 'reasoning_content') and chunk.reasoning_content:
                reasoning_content += chunk.reasoning_content
            
            if hasattr(chunk, 'response_metadata') and chunk.response_metadata:
                usage_data = chunk.response_metadata.get('usage')
                if usage_data:
                    token_usage = {
                        "prompt_tokens": usage_data.get('prompt_tokens', 0),
                        "completion_tokens": usage_data.get('completion_tokens', 0),
                        "total_tokens": usage_data.get('total_tokens', 0)
                    }

        total_time = time.time() - total_start_time
        first_token_delay = (first_token_time - total_start_time) if first_token_time else 0
        
        usage_info = ""
        if token_usage:
            usage_info = f", prompt_tokens: {token_usage['prompt_tokens']}, completion_tokens: {token_usage['completion_tokens']}"
        
        logger.info(f"analys_result completed - First token delay: {first_token_delay:.2f}s, Total time: {total_time:.2f}s, Tokens: {token_count}, Chunks: {chunk_count}{usage_info}")

        # 记录 Trace
        trace = ChainTrace(
            session_id=session_id,
            user_input=user_content[:500],
            task_type=task_type,
            keywords=keywords,
            prompt=system_content[:500],
            response=result[:500],
            reasoning=reasoning_content[:500] if reasoning_content else None,
            total_time_ms=round(total_time * 1000, 2),
            token_usage=token_usage
        )
        trace_logger.info(json.dumps(trace.to_dict(), ensure_ascii=False))
        
        return result
    except Exception as e:
        elapsed = time.time() - total_start_time
        logger.error(f"analys_result failed - elapsed: {elapsed:.2f}s, error: {e}", exc_info=True)
        return f"[ERROR] analys_result failed: {e}"


# =============================================================================
# 流式分析（生成器版本，用于 FastAPI 流式响应）
# =============================================================================
def analys_result_stream(messages: list, session_id: str = "", task_type: str = "", keywords: dict = None):
    """
    步骤4：调用 DeepSeek 大模型进行分析（流式输出，生成器版本）
    
    参数:
        messages: 消息列表（BaseMessage列表）
        session_id: 会话ID（用于追踪）
        task_type: 任务类型（如"故障诊断"、"知识查询"）
        keywords: 提取的关键词字典
    
    返回:
        生成器，逐块产生 {type, content} 或 {type, reasoning} 或 {type, end, ...} 字典
    """
    if keywords is None:
        keywords = {}
    total_start_time = time.time()
    logger.info(f"analys_result_stream called with {len(messages)} messages")

    system_content = ""
    user_content = ""
    for msg in messages:
        if hasattr(msg, 'type') and msg.type == 'system':
            system_content = getattr(msg, 'content', '')
        elif hasattr(msg, 'type') and msg.type == 'human':
            user_content = getattr(msg, 'content', '')
    
    logger.debug(f"analys_result_stream system prompt: {system_content[:200]}...")
    logger.info(f"analys_result_stream - {len(messages)} messages, system prompt: {system_content[:100]}...")
    
    full_prompt = f"System: {system_content}\n\nUser: {user_content}"

    from config.settings import LOG_LEVEL
    if LOG_LEVEL.upper() == "DEBUG":
        messages_serialized = []
        for msg in messages:
            msg_dict = {
                "type": getattr(msg, 'type', ''),
                "content": getattr(msg, 'content', '')
            }
            messages_serialized.append(msg_dict)
        
        prompt_logger.info(json.dumps({
            "prompt_length": len(full_prompt),
            "message_count": len(messages),
            "system_length": len(system_content),
            "user_length": len(user_content),
            "full_messages": messages_serialized,
            "full_system_prompt": system_content,
            "full_user_prompt": user_content
        }, ensure_ascii=False))
    else:
        prompt_logger.info(json.dumps({
            "prompt_length": len(full_prompt),
            "message_count": len(messages),
            "system_length": len(system_content),
            "user_length": len(user_content)
        }, ensure_ascii=False))

    from config.settings import DEEPSEEK_API_KEY
    raw_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

    openai_messages = []
    for msg in messages:
        msg_type = getattr(msg, 'type', '')
        msg_content = getattr(msg, 'content', '')
        if msg_type == 'system':
            openai_messages.append({"role": "system", "content": msg_content})
        elif msg_type == 'human':
            openai_messages.append({"role": "user", "content": msg_content})
        elif msg_type == 'ai':
            openai_messages.append({"role": "assistant", "content": msg_content})

    try:
        logger.info("Calling DeepSeek model with streaming...")

        result = ""
        reasoning_content = ""
        chunk_count = 0
        token_count = 0
        first_token_time = None
        
        token_usage = {}

        response = raw_client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=openai_messages,
            stream=True,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        )

        for chunk in response:
            chunk_count += 1
            if not chunk.choices:
                continue
            
            delta = chunk.choices[0].delta
            delta_content = getattr(delta, 'content', None)
            delta_reasoning = getattr(delta, 'reasoning_content', None)
        
            if delta_content:
                if not first_token_time:
                    first_token_time = time.time()
                token_count += 1
                result += delta_content
                yield {"type": "content", "content": delta_content}
        
            if delta_reasoning:
                reasoning_content += delta_reasoning
                yield {"type": "reasoning", "reasoning": delta_reasoning}
        
            if hasattr(chunk, 'usage') and chunk.usage:
                token_usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens
                }

        total_time = time.time() - total_start_time
        first_token_delay = (first_token_time - total_start_time) if first_token_time else 0
        
        usage_info = ""
        if token_usage:
            usage_info = f", prompt_tokens: {token_usage['prompt_tokens']}, completion_tokens: {token_usage['completion_tokens']}"
        
        logger.info(f"analys_result_stream completed - First token delay: {first_token_delay:.2f}s, Total time: {total_time:.2f}s, Tokens: {token_count}, Chunks: {chunk_count}{usage_info}")

        trace = ChainTrace(
            session_id=session_id,
            user_input=user_content[:500],
            task_type=task_type,
            keywords=keywords,
            prompt=system_content[:500],
            response=result[:500],
            reasoning=reasoning_content[:500] if reasoning_content else None,
            total_time_ms=round(total_time * 1000, 2),
            token_usage=token_usage
        )
        trace_logger.info(json.dumps(trace.to_dict(), ensure_ascii=False))
        
        yield {"type": "end", "content": result, "reasoning": reasoning_content, "token_usage": token_usage}
    except Exception as e:
        elapsed = time.time() - total_start_time
        logger.error(f"analys_result_stream failed - elapsed: {elapsed:.2f}s, error: {e}", exc_info=True)
        yield {"type": "error", "content": f"[ERROR] analys_result_stream failed: {e}"}


# =============================================================================
# LCEL 链组装
# =============================================================================

# LCEL 链 1：提取原始用户输入
original_input = RunnableParallel(
    original_user_input=lambda x: x["user_input"] if isinstance(x, dict) and "user_input" in x else x
)

# LCEL 链 2：预处理链（获取对话历史 → 统一理解（task_type 分类 + keyword 提取 + 问题 Enrich））
analysis_preprocessing_chain = (
    original_input
    | RunnableLambda(lambda x, config=None: get_history_for_preprocessing(x, config))
    | RunnableLambda(lambda x: unified_understand_wrapper({
          "user_input": x["original_user_input"],
          "chat_history": x.get("chat_history", [])
      }))
)

# LCEL 链 3：核心链（检索）
# 检索触发条件：
#   - RAG_ENABLED 为 True（总开关）
#   - task_type 为 "故障诊断" 或 "知识查询"（只有这两类需要外部知识）
memory_core_chain = (
    RunnablePassthrough.assign(
        reference_docs=lambda x: (
            retrieve_reference_docs(
                query=x.get("extract_result", {}).get("reconstructed_question", x.get("original_query", "")),
                battery_retriever=battery_retriever,
                manual_retriever=manual_retriever,
                retrieval_target=x.get("retrieval_target", "both"),
                keywords=x.get("extract_result", {}).get("keyword", {}),
                task_type="fault_diagnosis" if x.get("classify_result", {}).get("task_type", "") == "故障诊断" else "knowledge_query"
            ) if (RAG_ENABLED and x.get("classify_result", {}).get("task_type", "") in ("故障诊断", "知识查询"))
            else ""
        )
    )
    | RunnablePassthrough.assign(
        response=lambda x, config=None: analys_result(
            build_analys_prompt(x),
            session_id=config.get("configurable", {}).get("session_id", "") if config else "",
            task_type=x.get("classify_result", {}).get("task_type", ""),
            keywords=x.get("extract_result", {}).get("keyword", {})
        )
    )
).with_types(input_type=dict, output_type=dict)

# LCEL 链 4：分析链（生成）
final_analysis_chain = (
    analysis_preprocessing_chain
    | RunnableWithMessageHistory(
        memory_core_chain,
        get_session_history,
        input_messages_key="user_input",
        output_messages_key="response"
    )
)


if __name__ == "__main__":
    import sys
    set_debug(True)
    session_id = "test_session"
    print("幕墙助手已启动，输入问题开始对话（输入 exit 退出）")
    while True:
        try:
            user_input = input("\n>> ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input.strip():
            continue
        result = final_analysis_chain.invoke(
            {"user_input": user_input},
            {"configurable": {"session_id": session_id}}
        )
        print()