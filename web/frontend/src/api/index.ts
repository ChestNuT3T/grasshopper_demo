/**
 * API 封装模块：封装与 FastAPI 后端的所有接口调用
 * 
 * 接口列表：
 *   GET     /api/sessions                 → 获取所有会话列表
 *   POST    /api/sessions                 → 创建新会话
 *   GET     /api/sessions/{id}/messages   → 获取会话历史消息
 *   POST    /api/sessions/{id}/clear      → 清空会话消息
 *   DELETE  /api/sessions/{id}            → 删除会话
 *   POST    /api/chat/stream              → 流式聊天（核心端点，支持 AbortController 取消）
 *   POST    /api/messages/{id}/feedback   → 发送消息反馈
 */

import type { Session, Message, ChatRequest, StreamChunk } from '@/types';
import { fetchEventSource } from '@microsoft/fetch-event-source';

/** API 基础路径（通过 Vite 代理转发到后端） */
const API_BASE = '/api';

/**
 * 流式响应处理器接口
 * 
 * @interface StreamHandler
 * @property onChunk - 处理每个数据块（content/reasoning/end/error）
 * @property onError - 处理错误
 * @property onComplete - 处理完成
 * @property signal - AbortSignal，用于取消流式请求
 */
export interface StreamHandler {
  onChunk: (chunk: StreamChunk) => void;
  onError: (error: Error) => void;
  onComplete: () => void;
  signal?: AbortSignal;
}

/**
 * API 方法集合
 */
export const api = {
  /**
   * 获取所有会话列表
   * 
   * @returns Promise<{ sessions: Session[] }> - 会话列表
   */
  getSessions: async (): Promise<{ sessions: Session[] }> => {
    const response = await fetch(`${API_BASE}/sessions`);
    if (!response.ok) {
      throw new Error('Failed to fetch sessions');
    }
    return response.json();
  },

  /**
   * 获取会话历史消息
   * 
   * @param sessionId - 会话 ID
   * @returns Promise<{ messages: Message[] }> - 消息列表
   */
  getSessionMessages: async (sessionId: string): Promise<{ messages: Message[] }> => {
    const response = await fetch(`${API_BASE}/sessions/${sessionId}/messages`);
    if (!response.ok) {
      throw new Error('Failed to fetch messages');
    }
    return response.json();
  },

  /**
   * 创建新会话
   * 
   * @param name - 会话名称（可选）
   * @returns Promise<Session> - 新创建的会话
   */
  createSession: async (name: string): Promise<Session> => {
    const response = await fetch(`${API_BASE}/sessions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ name }),
    });
    if (!response.ok) {
      throw new Error('Failed to create session');
    }
    return response.json();
  },

  /**
   * 删除会话
   * 
   * @param sessionId - 会话 ID
   * @returns Promise<boolean> - 是否删除成功
   */
  deleteSession: async (sessionId: string): Promise<boolean> => {
    const response = await fetch(`${API_BASE}/sessions/${sessionId}`, {
      method: 'DELETE',
    });
    return response.ok;
  },

  /**
   * 清空会话消息
   * 
   * @param sessionId - 会话 ID
   * @returns Promise<boolean> - 是否清空成功
   */
  clearSessionMessages: async (sessionId: string): Promise<boolean> => {
    const response = await fetch(`${API_BASE}/sessions/${sessionId}/clear`, {
      method: 'POST',
    });
    return response.ok;
  },

  /**
   * 发送消息（流式响应）
   * 
   * 使用 SSE（Server-Sent Events）协议实时接收回答，
   * 通过 handler 回调处理流式数据。
   * 支持通过 handler.signal 传入 AbortSignal 来取消请求。
   * 
   * @param request - 聊天请求（user_input, session_id, rag_enabled）
   * @param handler - 流式响应处理器
   * @returns Promise<void>
   */
  sendMessage: async (
    request: ChatRequest,
    handler: StreamHandler
  ): Promise<void> => {
    const ctrl = new AbortController();
    let stopped = false;

    const stop = () => {
      if (stopped) return;
      stopped = true;
      ctrl.abort();
    };

    if (handler.signal) {
      if (handler.signal.aborted) {
        handler.onComplete();
        return;
      }
      handler.signal.addEventListener('abort', stop);
    }

    try {
      await fetchEventSource(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
        signal: ctrl.signal,
        onmessage: (event) => {
          try {
            const chunk: StreamChunk = JSON.parse(event.data);
            handler.onChunk(chunk);
            if (chunk.type === 'end' || chunk.type === 'error') {
              stopped = true;
              handler.onComplete();
              ctrl.abort();
            }
          } catch (e) {
            stopped = true;
            handler.onError(e as Error);
            ctrl.abort();
          }
        },
        onerror: (error) => {
          if (stopped || (error instanceof DOMException && error.name === 'AbortError')) {
            // Already handled
          } else {
            handler.onError(error);
          }
          throw error;
        },
        onclose: () => {
          if (!stopped) {
            handler.onComplete();
          }
        },
        openWhenHidden: true,
      });
    } catch (e) {
      // Errors are handled in onerror; abort errors are expected
    } finally {
      handler.signal?.removeEventListener('abort', stop);
    }
  },

  /**
   * 发送消息反馈
   * 
   * @param messageId - 消息 ID
   * @param feedback - 反馈类型（good: 有用 / bad: 无用）
   * @returns Promise<boolean> - 是否发送成功
   */
  sendFeedback: async (messageId: string, feedback: 'good' | 'bad'): Promise<boolean> => {
    const response = await fetch(`${API_BASE}/messages/${messageId}/feedback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ feedback }),
    });
    return response.ok;
  },
};
