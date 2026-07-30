/**
 * 状态管理模块：使用 Zustand 管理全局状态
 * 
 * 管理的状态包括：
 *   - 会话列表（sessions）
 *   - 当前会话（currentSessionId）
 *   - 消息列表（messages）
 *   - 加载状态（isTyping）
 *   - RAG 开关（ragEnabled）
 *   - 主题模式（theme）
 *   - 侧边栏状态（sidebarCollapsed）
 *   - 错误信息（error）
 */

import { create } from 'zustand';
import type { Session, Message, ChatState } from '@/types';
import { api } from '@/api';

/**
 * 聊天状态存储接口
 * 
 * 扩展 ChatState，添加操作方法
 */
interface ChatStore extends ChatState {
  /** 设置当前会话 */
  setCurrentSession: (sessionId: string) => void;
  
  /** 添加会话 */
  addSession: (session: Session) => void;
  
  /** 更新会话信息 */
  updateSession: (sessionId: string, updates: Partial<Session>) => void;
  
  /** 删除会话（本地状态） */
  deleteSession: (sessionId: string) => void;
  
  /** 设置消息列表 */
  setMessages: (messages: Message[]) => void;
  
  /** 添加消息 */
  addMessage: (message: Message) => void;
  
  /** 更新消息 */
  updateMessage: (messageId: string, updates: Partial<Message> | ((prev: Message) => Partial<Message>)) => void;
  
  /** 清空消息 */
  clearMessages: () => void;
  
  /** 设置加载状态 */
  setIsTyping: (isTyping: boolean) => void;
  
  /** 设置 RAG 开关 */
  setRagEnabled: (enabled: boolean) => void;
  
  /** 设置主题模式 */
  setTheme: (theme: 'light' | 'dark' | 'system') => void;
  
  /** 设置侧边栏折叠状态 */
  setSidebarCollapsed: (collapsed: boolean) => void;
  
  /** 设置错误信息 */
  setError: (error: string | null) => void;
  
  /** 从 API 加载会话列表 */
  loadSessions: () => Promise<void>;
  
  /** 从 API 加载会话消息 */
  loadMessages: (sessionId: string) => Promise<void>;
  
  /** 创建会话（调用 API） */
  createSession: (name: string) => Promise<Session | null>;
  
  /** 删除会话（调用 API） */
  deleteSessionApi: (sessionId: string) => Promise<boolean>;
  
  /** 清空会话消息（调用 API） */
  clearSessionMessages: (sessionId: string) => Promise<boolean>;
}

/**
 * 创建 Zustand 状态存储
 */
export const useChatStore = create<ChatStore>((set, get) => ({
  // =========================================================================
  // 初始状态
  // =========================================================================
  
  sessions: [],                      // 会话列表
  currentSessionId: 'default_session', // 当前会话 ID
  messages: [],                      // 当前会话的消息列表
  isTyping: false,                   // 是否正在生成回答
  ragEnabled: true,                  // RAG 检索是否启用
  theme: 'system',                   // 主题模式（light/dark/system）
  sidebarCollapsed: false,           // 侧边栏是否折叠
  error: null,                       // 错误信息

  // =========================================================================
  // 状态操作方法（同步）
  // =========================================================================

  /**
   * 设置当前会话 ID
   * 
   * @param sessionId - 会话 ID
   */
  setCurrentSession: (sessionId) => set({ currentSessionId: sessionId }),

  /**
   * 添加新会话到列表
   * 
   * @param session - 会话对象
   */
  addSession: (session) => set((state) => ({
    sessions: [...state.sessions, session],
  })),

  /**
   * 更新指定会话的信息
   * 
   * @param sessionId - 会话 ID
   * @param updates - 更新内容
   */
  updateSession: (sessionId, updates) => set((state) => ({
    sessions: state.sessions.map((s) =>
      s.id === sessionId ? { ...s, ...updates } : s
    ),
  })),

  /**
   * 删除会话（本地状态）
   * 如果删除的是当前会话，自动切换到其他会话
   * 
   * @param sessionId - 会话 ID
   */
  deleteSession: (sessionId) => set((state) => ({
    sessions: state.sessions.filter((s) => s.id !== sessionId),
    currentSessionId: state.currentSessionId === sessionId
      ? state.sessions.find((s) => s.id !== sessionId)?.id || 'default_session'
      : state.currentSessionId,
    messages: state.currentSessionId === sessionId ? [] : state.messages,
  })),

  /**
   * 设置消息列表
   * 
   * @param messages - 消息列表
   */
  setMessages: (messages) => set({ messages }),

  /**
   * 添加单条消息
   * 
   * @param message - 消息对象
   */
  addMessage: (message) => set((state) => ({
    messages: [...state.messages, message],
  })),

  /**
   * 更新指定消息（支持对象或函数式更新）
   * 
   * @param messageId - 消息 ID
   * @param updates - 更新内容（Partial<Message> 或 (prev: Message) => Partial<Message>）
   */
  updateMessage: (messageId: string, updates: Partial<Message> | ((prev: Message) => Partial<Message>)) => set((state) => ({
    messages: state.messages.map((m) => {
      if (m.id !== messageId) return m;
      const appliedUpdates = typeof updates === 'function' ? updates(m) : updates;
      return { ...m, ...appliedUpdates };
    }),
  })),

  /**
   * 清空消息列表
   */
  clearMessages: () => set({ messages: [] }),

  /**
   * 设置加载状态
   * 
   * @param isTyping - 是否正在生成
   */
  setIsTyping: (isTyping) => set({ isTyping }),

  /**
   * 设置 RAG 检索开关
   * 
   * @param enabled - 是否启用
   */
  setRagEnabled: (enabled) => set({ ragEnabled: enabled }),

  /**
   * 设置主题模式
   * 同时更新 DOM 的 dark 类名
   * 
   * @param theme - 主题模式（light/dark/system）
   */
  setTheme: (theme) => {
    set({ theme });
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else if (theme === 'light') {
      root.classList.remove('dark');
    } else {
      const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      if (systemDark) {
        root.classList.add('dark');
      } else {
        root.classList.remove('dark');
      }
    }
  },

  /**
   * 设置侧边栏折叠状态
   * 
   * @param collapsed - 是否折叠
   */
  setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),

  /**
   * 设置错误信息
   * 
   * @param error - 错误信息（null 表示清除）
   */
  setError: (error) => set({ error }),

  // =========================================================================
  // 异步操作方法（调用 API）
  // =========================================================================

  /**
   * 从 API 加载会话列表
   */
  loadSessions: async () => {
    try {
      const response = await api.getSessions();
      set({ sessions: response.sessions });
    } catch (error) {
      console.error('Failed to load sessions:', error);
      set({ sessions: [] });
    }
  },

  /**
   * 从 API 加载指定会话的消息
   * 
   * @param sessionId - 会话 ID
   */
  loadMessages: async (sessionId) => {
    try {
      const response = await api.getSessionMessages(sessionId);
      set({ messages: response.messages, currentSessionId: sessionId });
    } catch (error) {
      console.error('Failed to load messages:', error);
      set({ messages: [], currentSessionId: sessionId });
    }
  },

  /**
   * 创建新会话（调用 API）
   * 
   * @param name - 会话名称
   * @returns Promise<Session | null> - 创建的会话或 null（失败时）
   */
  createSession: async (name) => {
    try {
      const session = await api.createSession(name);
      if (session) {
        set((state) => ({
          sessions: [...state.sessions, session],
          currentSessionId: session.id,
          messages: [],
        }));
      }
      return session;
    } catch (error) {
      console.error('Failed to create session:', error);
      return null;
    }
  },

  /**
   * 删除会话（调用 API）
   * 
   * @param sessionId - 会话 ID
   * @returns Promise<boolean> - 是否删除成功
   */
  deleteSessionApi: async (sessionId) => {
    try {
      const success = await api.deleteSession(sessionId);
      if (success) {
        get().deleteSession(sessionId);
      }
      return success;
    } catch (error) {
      console.error('Failed to delete session:', error);
      return false;
    }
  },

  /**
   * 清空会话消息（调用 API）
   * 
   * @param sessionId - 会话 ID
   * @returns Promise<boolean> - 是否清空成功
   */
  clearSessionMessages: async (sessionId) => {
    try {
      const success = await api.clearSessionMessages(sessionId);
      if (success) {
        set({ messages: [] });
      }
      return success;
    } catch (error) {
      console.error('Failed to clear messages:', error);
      return false;
    }
  },
}));
