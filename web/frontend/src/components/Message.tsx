/**
 * 消息组件：DeepSeek 风格扁平化消息渲染
 * 
 * 设计特点：
 *   - AI 消息：纯文本，无气泡背景
 *   - 用户消息：浅灰圆角背景，右对齐
 *   - 支持 Markdown 渲染
 *   - 思考过程折叠展示（思考中自动展开，完成后自动折叠）
 *   - 检索 Trace 展示
 */

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight, oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { RefreshCw, Copy, Check } from 'lucide-react';
import { useState } from 'react';
import type { Message as MessageType } from '@/types';
import { ThinkingBlock } from './ThinkingBlock';
import { TraceBlock } from './TraceBlock';

interface MessageProps {
  message: MessageType;
  onRegenerate?: () => void;
  isThinking?: boolean;
  isStreaming?: boolean;
}

export function Message({ message, onRegenerate, isThinking = false, isStreaming = false }: MessageProps) {
  const isUser = message.role === 'user';
  const isDark = document.documentElement.classList.contains('dark');
  const [copied, setCopied] = useState(false);
  const [userCopied, setUserCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // fallback
    }
  };

  const handleCopyUser = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setUserCopied(true);
      setTimeout(() => setUserCopied(false), 2000);
    } catch {
      // fallback
    }
  };

  return (
    <div className={`message-row ${isUser ? 'user' : ''} animate-slide-up group`}>
      <div className={`message-content ${isUser ? 'text-right' : 'assistant'}`}>
        {/* 角色标签 */}
        <div className={`role-label ${isUser ? '' : 'assistant'}`}>
          {isUser ? '你' : 'AI'}
        </div>

        {/* 思考过程（仅 AI 消息，有思考内容或思考中时显示） */}
        {!isUser && (isThinking || (message.reasoning && message.reasoning.length > 0) || message.thinkingTime) && (
          <ThinkingBlock
            content={message.reasoning || ''}
            isThinking={isThinking}
            thinkingTime={message.thinkingTime || 0}
          />
        )}

        {/* 等待首次响应时的打字指示器（尚无思考内容和正文） */}
        {!isUser && isStreaming && !message.content && (!message.reasoning || message.reasoning.length === 0) && (
          <div className="typing-indicator">
            <span className="typing-dot" />
            <span className="typing-dot" />
            <span className="typing-dot" />
          </div>
        )}

        {/* 消息正文 */}
        {message.content && (
          <div className={isUser ? 'user-bubble text-[15px] leading-relaxed' : 'text-[15px] leading-relaxed'}>
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                code({ node, className, children, ...props }) {
                const isInline = !node?.tagName;
                const match = /language-(\w+)/.exec(className || '');
                const codeContent = String(children).replace(/\n$/, '');
                const plainLanguages = ['text', 'plaintext', 'plain', ''];

                if (!isInline && match && !plainLanguages.includes(match[1])) {
                  return (
                    <div className="code-block-wrapper">
                      <div className="code-block-header">
                        <span>{match[1]}</span>
                        <button
                          onClick={() => {
                            navigator.clipboard.writeText(codeContent);
                          }}
                          className="text-xs text-text-muted hover:text-text-main dark:hover:text-text-mainDark transition-colors"
                        >
                          复制
                        </button>
                      </div>
                      <SyntaxHighlighter
                        style={isDark ? oneDark : oneLight}
                        language={match[1]}
                        PreTag="div"
                        showLineNumbers={true}
                        wrapLines={true}
                        customStyle={{
                          margin: 0,
                          borderRadius: 0,
                          fontSize: '13px',
                        }}
                      >
                        {codeContent}
                      </SyntaxHighlighter>
                    </div>
                  );
                }

                if (!isInline && match) {
                  return (
                    <div className="code-block-wrapper">
                      <div className="code-block-header">
                        <button
                          onClick={() => {
                            navigator.clipboard.writeText(codeContent);
                          }}
                          className="text-xs text-text-muted hover:text-text-main dark:hover:text-text-mainDark transition-colors ml-auto"
                        >
                          复制
                        </button>
                      </div>
                      <pre className="p-3 overflow-x-auto text-[13px] font-mono leading-relaxed whitespace-pre bg-[#f6f8fa] dark:bg-[#1a1a1a] text-text-main dark:text-text-mainDark">
                        <code>{codeContent}</code>
                      </pre>
                    </div>
                  );
                }

                  return (
                    <code
                      className="px-1.5 py-0.5 rounded text-[13px] font-mono bg-[#f3f4f6] dark:bg-[#2a2a2a] text-text-main dark:text-text-mainDark"
                      {...props}
                    >
                      {children}
                    </code>
                  );
                },
                a({ href, children }) {
                  return (
                    <a
                      href={href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary hover:underline"
                    >
                      {children}
                    </a>
                  );
                },
                strong({ children }) {
                  return <strong className="font-bold italic">{children}</strong>;
                },
                ul({ children }) {
                  return <ul className="list-disc list-inside space-y-0.5 my-1.5">{children}</ul>;
                },
                ol({ children }) {
                  return <ol className="list-decimal list-inside space-y-0.5 my-1.5">{children}</ol>;
                },
                p({ children }) {
                  return <p className="my-[0.4em]">{children}</p>;
                },
                h1({ children }) {
                  return <h1 className="text-lg font-bold not-italic my-2.5">{children}</h1>;
                },
                h2({ children }) {
                  return <h2 className="text-base font-semibold not-italic my-2">{children}</h2>;
                },
                h3({ children }) {
                  return <h3 className="text-[15px] font-medium not-italic my-1.5">{children}</h3>;
                },
                h4({ children }) {
                  return <h4 className="text-sm font-semibold not-italic my-1">{children}</h4>;
                },
                blockquote({ children }) {
                  return (
                    <blockquote className="border-l-2 border-primary/40 pl-4 my-2 text-text-sub dark:text-text-subDark italic">
                      {children}
                    </blockquote>
                  );
                },
                table({ children }) {
                  return (
                    <div className="overflow-x-auto my-2">
                      <table className="border-collapse w-full text-sm">{children}</table>
                    </div>
                  );
                },
                th({ children }) {
                  return (
                    <th className="border border-border dark:border-border-dark px-3 py-2 bg-card dark:bg-card-dark font-semibold not-italic text-left whitespace-nowrap">
                      {children}
                    </th>
                  );
                },
                td({ children }) {
                  return (
                    <td className="border border-border dark:border-border-dark px-3 py-2">
                      {children}
                    </td>
                  );
                },
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        )}

        {/* 流式输出中，正文尚未开始但已有思考内容时，不显示额外指示器 */}

        {/* 检索 Trace（仅 AI 消息） */}
        {!isUser && message.trace && Object.keys(message.trace).length > 0 && (
          <TraceBlock trace={message.trace} />
        )}

        {/* 用户消息复制按钮（气泡下方，右对齐） */}
        {isUser && message.content && (
          <div className="flex justify-end mt-1">
            <button
              onClick={handleCopyUser}
              className="opacity-0 group-hover:opacity-100 flex items-center gap-1 px-2 py-1 rounded text-xs text-text-muted hover:text-text-main dark:hover:text-text-mainDark hover:bg-hover dark:hover:bg-hover-dark transition-all"
              title={userCopied ? '已复制' : '复制'}
            >
              {userCopied ? <Check size={12} /> : <Copy size={12} />}
              <span>{userCopied ? '已复制' : '复制'}</span>
            </button>
          </div>
        )}

        {/* 操作栏（仅 AI 消息，且非流式输出中） */}
        {!isUser && message.content && !isStreaming && (
          <div className="flex items-center gap-2 mt-2">
            <button
              onClick={handleCopy}
              className="action-btn"
              title="复制"
            >
              {copied ? <Check size={13} /> : <Copy size={13} />}
              <span>{copied ? '已复制' : '复制'}</span>
            </button>
            {onRegenerate && (
              <button
                onClick={onRegenerate}
                className="action-btn"
                title="重新生成"
              >
                <RefreshCw size={13} />
                <span>重新生成</span>
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
