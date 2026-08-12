"""
=============================================================================
        FastAPI 后端服务：Grasshopper 技术顾问的 API 接口
        完全复用现有核心模块，与 Streamlit 应用共享会话历史存储。
=============================================================================

调用流程：
    1. get_history_for_preprocessing(session_id) → 获取历史消息（文件存储）
    2. unified_understand_wrapper() → 意图理解和关键词提取
    3. retrieve_reference_docs() → RAG检索（条件触发）
    4. build_analys_prompt() → 构建完整消息列表
    5. analys_result_stream() → 流式生成回答
    6. get_session_history().add_message() → 保存历史到文件

API 接口列表：
    GET     /api/sessions                 → 获取所有会话列表
    POST    /api/sessions                 → 创建新会话
    GET     /api/sessions/{id}/messages   → 获取会话历史消息
    POST    /api/sessions/{id}/clear      → 清空会话消息
    DELETE  /api/sessions/{id}            → 删除会话
    POST    /api/chat/stream              → 流式聊天（核心端点）
    POST    /api/messages/{id}/feedback   → 发送消息反馈

注意事项：
    当前 API 采用直接函数调用方式，与 LCEL 链并存，后续可统一迁移。
    检索器实例在 init_retrievers() 中初始化后会同步到 src.core.chain 模块。
"""

import json
import uuid
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage

# 核心模块导入（与 Streamlit 应用一致）
from src.core.chain import (
    get_history_for_preprocessing,
    unified_understand_wrapper,
    build_analys_prompt,
    analys_result_stream,
)
from src.modules.retrieval import (
    retrieve_reference_docs,
    load_battery_retriever,
    load_manual_retriever,
)
from src.core.tools import get_session_history, get_chat_history
from src.core.chat_history import list_sessions, delete_session_record
from src.core.client import create_deepseek_client, get_embeddings
from src.core.logger import logger
from config.settings import (
    RAG_ENABLED,
    CHROMA_DIR,
    RETRIEVAL_CONFIG,
)


# =============================================================================
# 应用初始化
# =============================================================================

app = FastAPI(title="Grasshopper 技术顾问 API", version="0.1.0")

# CORS 配置：允许所有来源跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局检索器实例（应用启动时初始化）
battery_retriever = None
manual_retriever = None


def init_retrievers():
    """
    初始化 RAG 检索器（与 Streamlit 应用的 init_rag() 逻辑一致）
    
    创建电池手册检索器和官方手册检索器，供后续检索请求使用。
    成功初始化后，会将检索器实例同步到 src.core.chain 模块中的同名全局变量。
    失败时记录日志但不影响应用启动（RAG 功能降级）。
    """
    global battery_retriever, manual_retriever
    try:
        logger.info("Initializing RAG components...")
        persist_dir = str(CHROMA_DIR)
        
        # 获取嵌入模型和 LLM 客户端
        embeddings = get_embeddings()
        llm_client = create_deepseek_client(model="deepseek-v4-pro", streaming=False, timeout=120)
        
        # 从配置读取检索参数
        top_k = RETRIEVAL_CONFIG.get("battery_top_k", 10)
        logger.info(f"Loading retrievers with top_k={top_k}")
        
        # 加载检索器
        battery_retriever = load_battery_retriever(persist_dir, embeddings, llm_client, top_k)
        manual_retriever = load_manual_retriever(persist_dir, embeddings, top_k)
        
        # 同步赋值给 src.core.chain 模块中的同名变量
        import src.core.chain as chain_module
        chain_module.battery_retriever = battery_retriever
        chain_module.manual_retriever = manual_retriever
        
        logger.info("RAG components loaded successfully")
    except Exception as e:
        logger.error(f"检索器初始化失败: {e}")


# 应用启动时初始化检索器
init_retrievers()


# =============================================================================
# 会话管理接口
# =============================================================================

@app.get("/api/sessions")
def get_sessions():
    """
    获取所有会话列表（从 SQLite 读取）

    返回:
        {"sessions": [{id, name, messageCount, lastModified}, ...]}
    """
    session_list = list_sessions()

    # 如果没有会话，返回默认会话
    if not session_list:
        session_list.append({
            "id": "default_session",
            "name": "默认会话",
            "messageCount": 0,
            "lastModified": 0,
        })

    return {"sessions": session_list}


@app.post("/api/sessions")
def create_session(body: Dict = Body(...)):
    """
    创建新会话（SQLite 写入 sessions 表）

    参数:
        name: 会话名称（可选），不提供则自动生成 UUID 前8位

    返回:
        {"id": session_id, "name": session_name, "messageCount": 0, "lastModified": 0}
    """
    name = body.get("name")
    session_id = str(uuid.uuid4())[:8] if name is None else name.replace(" ", "_")

    # 创建会话记录（SQLiteChatMessageHistory 构造时自动写入 sessions 表）
    get_session_history(session_id)
    logger.info(f"Created new session: {session_id}")

    return {
        "id": session_id,
        "name": name or session_id.replace("_", " ").title(),
        "messageCount": 0,
        "lastModified": 0,
    }


@app.get("/api/sessions/{session_id}/messages")
def get_session_messages(session_id: str):
    """
    获取会话历史消息（从 SQLite 读取）

    参数:
        session_id: 会话 ID

    返回:
        {"messages": [{id, role, content, reasoning, trace, createdAt}, ...]}
    """
    try:
        messages = get_chat_history(session_id, format_output=False)
        formatted_messages = []
        for i, msg in enumerate(messages):
            role = 'user' if hasattr(msg, 'type') and msg.type == 'human' else 'assistant'
            content = getattr(msg, 'content', '')
            additional_kwargs = getattr(msg, 'additional_kwargs', {}) or {}
            reasoning = additional_kwargs.get('reasoning', '')
            trace = additional_kwargs.get('trace')
            created_at = additional_kwargs.get('created_at', 0)
            formatted_messages.append({
                "id": f"msg_{i}",
                "role": role,
                "content": content,
                "reasoning": reasoning,
                "trace": trace,
                "createdAt": created_at,
            })
        return {"messages": formatted_messages}
    except Exception as e:
        logger.error(f"Failed to get messages for session {session_id}: {e}")
        return {"messages": []}


@app.post("/api/sessions/{session_id}/clear")
def clear_session_messages(session_id: str):
    """
    清空会话消息
    
    参数:
        session_id: 会话 ID
    
    返回:
        {"success": True/False, "error": 错误信息（可选）}
    """
    try:
        history = get_session_history(session_id)
        history.clear()
        logger.info(f"Cleared messages for session: {session_id}")
        return {"success": True}
    except Exception as e:
        logger.error(f"Failed to clear messages for session {session_id}: {e}")
        return {"success": False, "error": str(e)}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    """
    删除会话（SQLite 级联删除会话记录与消息）

    参数:
        session_id: 会话 ID

    返回:
        {"success": True/False, "error": 错误信息（可选）}
    """
    try:
        if delete_session_record(session_id):
            logger.info(f"Deleted session: {session_id}")
            return {"success": True}
        raise HTTPException(status_code=404, detail="会话不存在")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# 流式聊天核心逻辑
# =============================================================================

def generate_stream_response(user_input: str, session_id: str, rag_enabled: bool):
    """
    生成流式响应的生成器函数（核心业务逻辑）
    
    参数:
        user_input: 用户输入的问题
        session_id: 会话 ID，用于获取历史消息和保存结果
        rag_enabled: 是否启用 RAG 检索增强
    
    返回:
        生成器，逐块产生 SSE 格式的字符串（data: {...}\n\n）
        块类型：
            - "content": 回答内容块
            - "reasoning": 思考过程块
            - "end": 完成标志（含完整内容和思考过程）
            - "error": 错误信息
    """
    full_content = ""
    full_reasoning = ""
    
    try:
        logger.info(f"Processing chat request - session: {session_id}, input: {user_input[:50]}...")
        
        # 步骤1：获取历史消息用于预处理
        preprocessed = get_history_for_preprocessing(
            {"user_input": user_input},
            {"configurable": {"session_id": session_id}}
        )
        
        # 步骤2：统一理解（意图分类 + 关键词提取 + 问题丰富）
        understand_result = unified_understand_wrapper({
            "user_input": user_input,
            "chat_history": preprocessed.get("chat_history", []),
        })
        
        # 提取理解结果中的关键信息
        task_type = understand_result.get("classify_result", {}).get("task_type", "")
        retrieval_target = understand_result.get("retrieval_target", "both")
        reconstructed_question = understand_result.get(
            "extract_result", {}
        ).get("reconstructed_question", user_input)
        keywords = understand_result.get("extract_result", {}).get("keyword", {})
        
        # 步骤3：RAG 检索（条件触发）
        reference_docs = ""
        if rag_enabled and RAG_ENABLED and task_type in ("故障诊断", "知识查询"):
            # 增加判空逻辑：若检索器为 None，则跳过 RAG 检索
            if battery_retriever is None or manual_retriever is None:
                logger.warning("RAG retrieval skipped: battery_retriever or manual_retriever is None")
            else:
                logger.info(f"RAG enabled, retrieving docs for task: {task_type}")
                reference_docs = retrieve_reference_docs(
                    query=reconstructed_question,
                    battery_retriever=battery_retriever,
                    manual_retriever=manual_retriever,
                    retrieval_target=retrieval_target,
                    keywords=keywords,
                    task_type="fault_diagnosis" if task_type == "故障诊断" else "knowledge_query"
                )
                logger.info(f"Retrieved {len(reference_docs) if reference_docs else 0} chars of reference docs")
        
        # 步骤4：构建完整 Prompt
        data = {
            **understand_result,
            "reference_docs": reference_docs,
            "original_query": user_input,
        }
        
        messages = build_analys_prompt(data)
        logger.info(f"Built prompt with {len(messages)} messages")
        
        # 步骤5：流式生成回答
        for chunk in analys_result_stream(
            messages=messages,
            session_id=session_id,
            task_type=task_type,
            keywords=keywords
        ):
            chunk_type = chunk.get("type")
            
            if chunk_type == "content":
                content = chunk.get("content", "")
                full_content += content
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            
            elif chunk_type == "reasoning":
                reasoning = chunk.get("reasoning", "")
                full_reasoning += reasoning
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            
            elif chunk_type == "end":
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                
                # 步骤6：保存历史消息到文件（包含 reasoning）
                try:
                    history = get_session_history(session_id)
                    history.add_message(HumanMessage(content=user_input))
                    history.add_message(AIMessage(
                        content=full_content,
                        additional_kwargs={"reasoning": full_reasoning}
                    ))
                    logger.info(f"Saved messages to session: {session_id}")
                except Exception as save_error:
                    logger.error(f"Failed to save messages: {save_error}")
                break
            
            elif chunk_type == "error":
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                break
    
    except Exception as e:
        logger.error(f"Stream generation failed: {e}", exc_info=True)
        # 区分系统错误和业务错误，给用户友好提示
        error_type = type(e).__name__
        if "LLMServiceError" in error_type or "Service" in error_type:
            user_msg = "AI 服务暂时不可用，请稍后重试。"
        elif "LLMOutputError" in error_type:
            user_msg = "AI 响应格式异常，请重新描述您的问题。"
        else:
            user_msg = f"生成响应时出错：{str(e)}"
        error_chunk = {"type": "error", "content": user_msg}
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"


# =============================================================================
# Pydantic 数据模型
# =============================================================================

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """
    流式聊天请求的数据模型
    
    参数:
        user_input: 用户输入的问题（必填）
        session_id: 会话 ID（默认: "default_session"）
        rag_enabled: 是否启用 RAG 检索增强（默认: True）
    """
    user_input: str
    session_id: str = "default_session"
    rag_enabled: bool = True


# =============================================================================
# 流式聊天接口（核心端点）
# =============================================================================

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式聊天接口（核心端点）
    
    接收用户输入，经过理解、检索（可选）、生成等步骤，
    通过 SSE（Server-Sent Events）协议实时返回回答。
    
    请求体 JSON:
        {
            "user_input": "用户问题",
            "session_id": "会话ID",
            "rag_enabled": true/false
        }
    
    响应格式（SSE）:
        data: {"type": "content", "content": "回答内容"}
        data: {"type": "reasoning", "reasoning": "思考过程"}
        data: {"type": "end", "content": "完整回答", "reasoning": "完整思考"}
        data: {"type": "error", "content": "错误信息"}
    """
    if not request.user_input.strip():
        raise HTTPException(status_code=400, detail="输入不能为空")
    
    logger.info(f"Received chat request - session: {request.session_id}, input: {request.user_input[:50]}...")
    
    return StreamingResponse(
        generate_stream_response(request.user_input, request.session_id, request.rag_enabled),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# 消息反馈接口
# =============================================================================

@app.post("/api/messages/{message_id}/feedback")
def send_feedback(message_id: str, feedback: str = Query(..., description="反馈类型: good/bad")):
    """
    发送消息反馈
    
    参数:
        message_id: 消息 ID
        feedback: 反馈类型（good: 有用 / bad: 无用）
    
    返回:
        {"success": True}
    """
    logger.info(f"消息反馈: {message_id} - {feedback}")
    return {"success": True}


# =============================================================================
# 应用入口
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
