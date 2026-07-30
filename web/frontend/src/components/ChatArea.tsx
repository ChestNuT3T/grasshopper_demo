/**
 * 聊天区域组件：DeepSeek 风格扁平化聊天界面
 * 
 * 功能：
 *   - 消息列表渲染
 *   - 流式显示 AI 回答（思维链实时流式输出 → 正文流式输出）
 *   - 思考过程展示（思考中自动展开，完成后自动折叠）
 *   - 自动滚动到底部
 *   - 错误处理
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { useChatStore } from '@/store/useChatStore';
import { api } from '@/api';
import { Message } from './Message';
import { InputArea } from './InputArea';
import { Bot } from 'lucide-react';

export function ChatArea() {
  const {
    messages,
    currentSessionId,
    isTyping,
    ragEnabled,
    error,
    addMessage,
    updateMessage,
    setIsTyping,
    setError,
    setMessages,
  } = useChatStore();

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const thinkingStartRef = useRef<number | null>(null);
  const thinkingEndRef = useRef<number | null>(null);
  const isThinkingPhaseRef = useRef(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);
  const [isThinkingPhase, setIsThinkingPhase] = useState(false);

  const handleStop = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsTyping(false);
    setStreamingMessageId(null);
    setIsThinkingPhase(false);
    isThinkingPhaseRef.current = false;
    setError(null);
  }, [setIsTyping, setError]);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const handleSendMessage = async (userInput: string) => {
    setError(null);

    const userMessage = {
      id: `msg_${Date.now()}`,
      role: 'user' as const,
      content: userInput,
      createdAt: Date.now(),
    };

    addMessage(userMessage);
    setIsTyping(true);
    setIsThinkingPhase(true);
    isThinkingPhaseRef.current = true;

    const now = Date.now();
    thinkingStartRef.current = now;
    thinkingEndRef.current = null;

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    const tempAssistantMessage = {
      id: `msg_${Date.now()}_temp`,
      role: 'assistant' as const,
      content: '',
      reasoning: '',
      createdAt: Date.now(),
    };
    setStreamingMessageId(tempAssistantMessage.id);
    addMessage(tempAssistantMessage);

    try {
      await api.sendMessage(
        {
          user_input: userInput,
          session_id: currentSessionId,
          rag_enabled: ragEnabled,
        },
        {
          onChunk: (chunk) => {
            if (chunk.type === 'reasoning' && chunk.reasoning) {
              updateMessage(tempAssistantMessage.id, (prev) => ({
                reasoning: (prev.reasoning || '') + chunk.reasoning,
              }));
            } else if (chunk.type === 'content' && chunk.content) {
              if (isThinkingPhaseRef.current) {
                isThinkingPhaseRef.current = false;
                setIsThinkingPhase(false);
                const endTime = Date.now();
                thinkingEndRef.current = endTime;
              }
              updateMessage(tempAssistantMessage.id, (prev) => ({
                content: (prev.content || '') + chunk.content,
              }));
            } else if (chunk.type === 'end') {
              const startTime = thinkingStartRef.current;
              const endTime = thinkingEndRef.current || Date.now();
              if (!thinkingEndRef.current) {
                thinkingEndRef.current = endTime;
              }
              const finalThinkingTime = startTime
                ? (endTime - startTime) / 1000
                : 0;

              updateMessage(tempAssistantMessage.id, (prev) => ({
                id: `msg_${Date.now()}`,
                content: prev.content || '',
                reasoning: prev.reasoning || '',
                trace: chunk.trace,
                thinkingTime: finalThinkingTime,
              }));
              setStreamingMessageId(null);
              setIsThinkingPhase(false);
              isThinkingPhaseRef.current = false;
            } else if (chunk.type === 'error') {
              setError(chunk.content || '未知错误');
              updateMessage(tempAssistantMessage.id, {
                content: chunk.content || '生成响应时出错',
              });
              setStreamingMessageId(null);
              setIsThinkingPhase(false);
              isThinkingPhaseRef.current = false;
            }
          },
          onError: (error) => {
            console.error('Stream error:', error);
            setError(error.message);
            updateMessage(tempAssistantMessage.id, {
              content: `出错了：${error.message}`,
            });
            setStreamingMessageId(null);
            setIsThinkingPhase(false);
            isThinkingPhaseRef.current = false;
          },
          onComplete: () => {
            setIsTyping(false);
            abortControllerRef.current = null;
          },
          signal: abortController.signal,
        }
      );
    } catch (error) {
      console.error('Send message error:', error);
      setError((error as Error).message);
      setIsTyping(false);
      setStreamingMessageId(null);
      setIsThinkingPhase(false);
      isThinkingPhaseRef.current = false;
      abortControllerRef.current = null;
    }
  };

  const handleRegenerate = async () => {
    if (messages.length < 2) return;
    const lastUserMessage = messages[messages.length - 2];
    setMessages(messages.slice(0, -1));
    await handleSendMessage(lastUserMessage.content);
  };

  return (
    <div className="flex-1 flex flex-col min-h-0 bg-bg dark:bg-bg-dark">
      {/* 顶部栏 */}
      <header className="h-12 flex items-center border-b border-border dark:border-border-dark bg-transparent flex-shrink-0" style={{ paddingLeft: 'var(--header-padding-left, 24px)', paddingRight: '24px' }}>
        <div className="flex items-center gap-2.5">
          <Bot size={18} className="text-primary" />
          <span className="text-[15px] font-semibold text-text-main dark:text-text-mainDark">
            Grasshopper 技术顾问
          </span>
        </div>
      </header>

      {/* 消息区域 */}
      <main className="flex-1 overflow-y-auto scrollbar-thin">
        {messages.length === 0 && !isTyping ? (
          /* 欢迎页 */
          <div className="flex flex-col items-center justify-center h-full px-4">
            <div className="w-16 h-16 rounded-2xl bg-primary-light dark:bg-primary/10 flex items-center justify-center mb-5">
              <Bot size={28} className="text-primary" />
            </div>
            <h2 className="text-xl font-semibold text-text-main dark:text-text-mainDark mb-2">
              Grasshopper 技术顾问
            </h2>
            <p className="text-sm text-text-sub dark:text-text-subDark max-w-md text-center leading-relaxed">
              幕墙参数化设计助手，擅长 Grasshopper、Rhino、Dynamo 等工具的使用与问题解答
            </p>
          </div>
        ) : (
          <div className="max-w-[1200px] mx-auto px-6 py-4">
            {/* 错误提示 */}
            {error && (
              <div className="error-bar animate-slide-up">
                <span>{error}</span>
                <button onClick={() => setError(null)}>关闭</button>
              </div>
            )}

            {/* 消息列表 */}
            {messages.map((message, index) => {
              const isLastAssistant = index === messages.length - 1 && message.role === 'assistant';
              const isThisStreaming = isLastAssistant && streamingMessageId === message.id;
              const msgThinking = isThisStreaming && isThinkingPhase;
              const msgStreaming = isThisStreaming && isTyping;

              return (
                <Message
                  key={message.id}
                  message={message}
                  onRegenerate={
                    index === messages.length - 1 &&
                    message.role === 'assistant' &&
                    !isTyping
                      ? handleRegenerate
                      : undefined
                  }
                  isThinking={msgThinking}
                  isStreaming={msgStreaming}
                />
              );
            })}

            <div ref={messagesEndRef} />
          </div>
        )}
      </main>

      {/* 输入区域 */}
      <InputArea onSend={handleSendMessage} isTyping={isTyping} onStop={handleStop} />
    </div>
  );
}
