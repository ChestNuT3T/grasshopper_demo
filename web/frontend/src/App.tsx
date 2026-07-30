/**
 * 应用主组件：DeepSeek 风格布局
 * 
 * 布局结构：
 *   - 左侧：Sidebar（可折叠）
 *   - 右侧：ChatArea（主聊天区域）
 *   - 侧边栏折叠时显示展开按钮（悬浮在左上角，不遮挡标题）
 */

import { useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatArea } from './components/ChatArea';
import { useChatStore } from './store/useChatStore';
import { PanelLeft } from 'lucide-react';

function App() {
  const {
    loadSessions,
    setTheme,
    theme,
    sessions,
    loadMessages,
    sidebarCollapsed,
    setSidebarCollapsed,
  } = useChatStore();

  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') as 'light' | 'dark' | 'system' | null;
    if (savedTheme) {
      setTheme(savedTheme);
    } else {
      setTheme('system');
    }
    loadSessions();
  }, [loadSessions, setTheme]);

  useEffect(() => {
    localStorage.setItem('theme', theme);
  }, [theme]);

  useEffect(() => {
    if (sessions.length > 0) {
      loadMessages(sessions[0].id);
    }
  }, [sessions, loadMessages]);

  useEffect(() => {
    document.documentElement.style.setProperty(
      '--header-padding-left',
      sidebarCollapsed ? '56px' : '24px'
    );
  }, [sidebarCollapsed]);

  return (
    <div className="h-screen flex bg-bg dark:bg-bg-dark text-text-main dark:text-text-mainDark">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0 relative">
        {/* 侧边栏折叠时显示展开按钮 - 放在header上方，避免与标题重叠 */}
        {sidebarCollapsed && (
          <button
            onClick={() => setSidebarCollapsed(false)}
            className="absolute top-2.5 left-3 z-20 w-7 h-7 flex items-center justify-center rounded-lg hover:bg-hover dark:hover:bg-hover-dark text-text-muted hover:text-text-main dark:hover:text-text-mainDark transition-colors bg-bg dark:bg-bg-dark shadow-sm border border-border dark:border-border-dark"
            title="展开侧边栏"
          >
            <PanelLeft size={16} />
          </button>
        )}
        <ChatArea />
      </div>
    </div>
  );
}

export default App;
