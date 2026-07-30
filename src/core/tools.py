"""工具模块：提供通用工具函数，包括 JSON 解析、历史记录管理等"""

import sys
import json
import re
import os
import time
from typing import Dict, Any, List

from langchain_core.messages import BaseMessage
from langchain_community.chat_message_histories import FileChatMessageHistory

from src.core.logger import logger


def clean_json_output(text: str) -> str:
    """
    清理模型输出的 JSON，移除可能存在的 Markdown 代码块标记
    
    参数:
        text: 模型输出的文本
        
    返回:
        清理后的 JSON 字符串
    """
    if not text:
        return text
    
    text = re.sub(r'^```(json)?\s*', '', text.strip(), flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text.strip())
    
    text = re.sub(r'^[^{]*', '', text.strip())
    
    brace_count = 0
    result = []
    in_string = False
    escape = False
    
    for char in text:
        if escape:
            result.append(char)
            escape = False
            continue
            
        if char == '\\':
            result.append(char)
            escape = True
            continue
            
        if char == '"':
            in_string = not in_string
            result.append(char)
            continue
            
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    result.append(char)
                    break
        
        result.append(char)
    
    return ''.join(result)


def parse_json_with_fallback(text: str, default_result: dict = None) -> dict:
    """
    解析 JSON，带有自动修复和降级处理
    
    参数:
        text: 待解析的文本
        default_result: 解析失败时返回的默认值
        
    返回:
        解析后的字典
    """
    if default_result is None:
        default_result = {}
    
    if not text or not text.strip():
        return default_result
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    try:
        cleaned = clean_json_output(text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    try:
        cleaned = clean_json_output(text)
        cleaned = re.sub(r'(?<!\\)"(?!:)', '\\"', cleaned)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    print(f"[WARNING] Failed to parse JSON, returning default: {text[:100]}...")
    return default_result


def get_chat_history(session_id: str = None, messages: List[BaseMessage] = None, format_output: bool = True) -> str:
    """
    获取或格式化聊天历史（统一接口）
    
    参数:
        session_id: 会话ID（与 messages 二选一）
        messages: 消息列表（与 session_id 二选一）
        format_output: 是否返回格式化字符串（True返回字符串，False返回消息列表）
        
    返回:
        格式化的历史消息字符串或原始消息列表
    """
    # 如果提供了 session_id，从文件读取历史
    if session_id:
        try:
            history = get_session_history(session_id)
            messages = history.messages if (history and history.messages) else []
        except Exception as e:
            print(f"[WARNING] Failed to get chat history: {e}")
            messages = []
    
    # 如果没有消息，返回空值
    if not messages:
        return "" if format_output else []
    
    # 如果需要格式化输出
    if format_output:
        formatted_lines = []
        for msg in messages:
            role = "用户" if hasattr(msg, 'type') and msg.type == 'human' else "助手"
            content = getattr(msg, 'content', '')
            content = content.strip()
            if content:
                formatted_lines.append(f"{role}: {content}")
        return "\n".join(formatted_lines)
    
    # 返回原始消息列表
    return messages


def safe_print(text: str):
    """
    安全打印，处理编码问题
    
    参数:
        text: 要打印的文本
    """
    try:
        encoded = text.encode('utf-8')
        sys.stdout.buffer.write(encoded)
        sys.stdout.flush()
    except Exception:
        try:
            safe_text = text.encode('utf-8', errors='replace').decode('utf-8')
            encoded = safe_text.encode('utf-8')
            sys.stdout.buffer.write(encoded)
            sys.stdout.flush()
        except Exception:
            try:
                ascii_text = text.encode('ascii', errors='replace').decode('ascii')
                print(ascii_text, end='', flush=True)
            except Exception:
                pass


def cleanup_old_history(history: FileChatMessageHistory, max_age_minutes: int = 30):
    """
    清理指定时间之前的历史记录
    
    参数:
        history: FileChatMessageHistory 对象
        max_age_minutes: 最大保留时间（分钟），默认为30分钟
    """
    if not history.messages:
        return

    cutoff_time = time.time() - (max_age_minutes * 60)

    filtered_messages = [
        msg for msg in history.messages
        if getattr(msg, 'additional_kwargs', {}).get('timestamp', time.time()) > cutoff_time
    ]

    history.clear()
    for msg in filtered_messages:
        history.add_message(msg)


def get_session_history(session_id: str) -> FileChatMessageHistory:
    """
    获取会话历史，存储在本地硬盘中
    
    参数:
        session_id: 会话ID
        
    返回:
        FileChatMessageHistory 对象
    """
    from config.settings import CHAT_HISTORY_DIR
    history_dir = str(CHAT_HISTORY_DIR)
    os.makedirs(history_dir, exist_ok=True)

    file_path = os.path.join(history_dir, f"{session_id}.json")

    history = FileChatMessageHistory(file_path)

    cleanup_old_history(history)

    logger.info(f"get_session_history - session_id: {session_id}, message_count: {len(history.messages)}")

    return history