/**
 * 侧边栏组件：DeepSeek 风格简洁侧边栏
 * 
 * 功能：
 *   - 会话列表
 *   - 创建/删除会话
 *   - 主题切换
 *   - RAG 开关
 *   - 折叠/展开
 */

import { useState } from 'react';
import {
  MessageSquare,
  Plus,
  Trash2,
  Sun,
  Moon,
  Monitor,
  Search,
  PanelLeftClose,
} from 'lucide-react';
import { useChatStore } from '@/store/useChatStore';
import type { Session } from '@/types';

export function Sidebar() {
  const {
    sessions,
    currentSessionId,
    sidebarCollapsed,
    theme,
    ragEnabled,
    setSidebarCollapsed,
    setTheme,
    setRagEnabled,
    setCurrentSession,
    loadMessages,
    deleteSessionApi,
    createSession,
  } = useChatStore();

  const [showDeleteConfirm, setShowDeleteConfirm] = useState<string | null>(null);

  const handleSessionClick = async (session: Session) => {
    if (session.id !== currentSessionId) {
      setCurrentSession(session.id);
      await loadMessages(session.id);
    }
  };

  const handleConfirmDelete = async (sessionId: string) => {
    await deleteSessionApi(sessionId);
    setShowDeleteConfirm(null);
  };

  const handleCreateSession = async () => {
    const name = prompt('请输入会话名称：', `会话 ${sessions.length + 1}`);
    if (name) {
      await createSession(name);
    }
  };

  return (
    <aside
      className={`border-r border-border dark:border-border-dark bg-sidebar dark:bg-sidebar-dark flex flex-col transition-all duration-200 ${
        sidebarCollapsed ? 'w-0 overflow-hidden border-r-0' : 'w-[280px]'
      }`}
    >
      {/* 顶部 */}
      <div className="h-12 flex items-center px-4 border-b border-border dark:border-border-dark flex-shrink-0">
        <span className="text-[15px] font-semibold text-text-main dark:text-text-mainDark flex items-center gap-2">
          <MessageSquare size={18} className="text-primary" />
          Grasshopper
        </span>
        <button
          onClick={() => setSidebarCollapsed(true)}
          className="ml-auto p-1 rounded-md hover:bg-hover dark:hover:bg-hover-dark text-text-muted hover:text-text-main dark:hover:text-text-mainDark transition-colors"
          title="收起侧边栏"
        >
          <PanelLeftClose size={16} />
        </button>
      </div>

      {/* 会话操作 */}
      <div className="px-3 py-2 flex-shrink-0">
        <button
          onClick={handleCreateSession}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg border border-border dark:border-border-dark text-text-sub dark:text-text-subDark hover:bg-hover dark:hover:bg-hover-dark hover:text-text-main dark:hover:text-text-mainDark transition-colors text-sm"
        >
          <Plus size={15} />
          <span>新建会话</span>
        </button>
      </div>

      {/* 会话列表 */}
      <div className="flex-1 overflow-y-auto scrollbar-thin px-2">
        {sessions.length === 0 ? (
          <div className="text-center py-8 text-text-muted text-sm">
            暂无会话
          </div>
        ) : (
          <div className="space-y-0.5">
            {sessions.map((session) => (
              <div
                key={session.id}
                className={`relative group flex items-center gap-2.5 px-3 py-2.5 rounded-lg cursor-pointer transition-colors ${
                  currentSessionId === session.id
                    ? 'bg-primary-light dark:bg-primary/10 text-primary'
                    : 'text-text-main dark:text-text-mainDark hover:bg-hover dark:hover:bg-hover-dark'
                }`}
                onClick={() => handleSessionClick(session)}
              >
                <MessageSquare size={15} className="flex-shrink-0" />
                <span className="flex-1 text-sm truncate">
                  {session.name}
                </span>
                <span className="text-xs text-text-muted flex-shrink-0">
                  {session.messageCount}
                </span>
                {/* 删除按钮 */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setShowDeleteConfirm(session.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-50 dark:hover:bg-red-900/20 text-text-muted hover:text-red-500 transition-all flex-shrink-0"
                >
                  <Trash2 size={13} />
                </button>

                {/* 删除确认 */}
                {showDeleteConfirm === session.id && (
                  <div className="absolute inset-0 bg-sidebar dark:bg-sidebar-dark rounded-lg flex items-center justify-center gap-2 z-10 shadow-sm border border-border dark:border-border-dark">
                    <span className="text-xs text-text-sub dark:text-text-subDark">确认删除？</span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setShowDeleteConfirm(null);
                      }}
                      className="px-2 py-0.5 text-xs rounded bg-hover dark:bg-hover-dark text-text-sub dark:text-text-subDark hover:text-text-main dark:hover:text-text-mainDark"
                    >
                      取消
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleConfirmDelete(session.id);
                      }}
                      className="px-2 py-0.5 text-xs rounded bg-red-500 text-white hover:bg-red-600"
                    >
                      删除
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 底部设置 */}
      <div className="border-t border-border dark:border-border-dark p-3 flex-shrink-0 space-y-2">
        {/* RAG 开关 - 带切换滑块样式 */}
        <button
          onClick={() => setRagEnabled(!ragEnabled)}
          className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg transition-all text-sm ${
            ragEnabled
              ? 'bg-primary/10 dark:bg-primary/15 text-primary border border-primary/20'
              : 'bg-hover/50 dark:bg-hover-dark/50 text-text-sub dark:text-text-subDark border border-transparent hover:bg-hover dark:hover:bg-hover-dark'
          }`}
        >
          <Search size={15} className={ragEnabled ? 'text-primary' : ''} />
          <span className="flex-1 text-left font-medium">RAG 检索增强</span>
          <div className={`relative w-9 h-5 rounded-full transition-colors ${
            ragEnabled ? 'bg-primary' : 'bg-gray-300 dark:bg-gray-600'
          }`}>
            <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow-sm transition-transform ${
              ragEnabled ? 'translate-x-4' : 'translate-x-0.5'
            }`} />
          </div>
        </button>

        {/* 主题切换 */}
        <div className="flex rounded-lg bg-hover dark:bg-hover-dark p-0.5">
          <button
            onClick={() => setTheme('light')}
            className={`flex-1 flex items-center justify-center py-1.5 rounded-md transition-colors ${
              theme === 'light'
                ? 'bg-white dark:bg-[#3a3a3a] text-text-main dark:text-text-mainDark shadow-sm'
                : 'text-text-muted hover:text-text-sub dark:hover:text-text-subDark'
            }`}
            title="浅色模式"
          >
            <Sun size={15} />
          </button>
          <button
            onClick={() => setTheme('dark')}
            className={`flex-1 flex items-center justify-center py-1.5 rounded-md transition-colors ${
              theme === 'dark'
                ? 'bg-white dark:bg-[#3a3a3a] text-text-main dark:text-text-mainDark shadow-sm'
                : 'text-text-muted hover:text-text-sub dark:hover:text-text-subDark'
            }`}
            title="深色模式"
          >
            <Moon size={15} />
          </button>
          <button
            onClick={() => setTheme('system')}
            className={`flex-1 flex items-center justify-center py-1.5 rounded-md transition-colors ${
              theme === 'system'
                ? 'bg-white dark:bg-[#3a3a3a] text-text-main dark:text-text-mainDark shadow-sm'
                : 'text-text-muted hover:text-text-sub dark:hover:text-text-subDark'
            }`}
            title="跟随系统"
          >
            <Monitor size={15} />
          </button>
        </div>
      </div>
    </aside>
  );
}