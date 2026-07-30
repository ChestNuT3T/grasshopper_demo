/**
 * 输入区域组件：DeepSeek 风格简洁输入框
 * 
 * 功能：
 *   - 多行文本输入（Enter 发送，Shift+Enter 换行）
 *   - 推荐问题快捷点击
 *   - 流式生成时显示停止按钮
 *   - 与上方消息区域左右对齐
 */

import { useState, useRef, useEffect } from 'react';
import { ArrowUp, Square } from 'lucide-react';
import { RECOMMENDED_QUESTIONS } from '@/types';

interface InputAreaProps {
  onSend: (message: string) => void;
  isTyping: boolean;
  onStop?: () => void;
}

export function InputArea({ onSend, isTyping, onStop }: InputAreaProps) {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`;
    }
  }, [input]);

  const handleSend = () => {
    if (input.trim() && !isTyping) {
      onSend(input.trim());
      setInput('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleStop = () => {
    if (onStop) {
      onStop();
    }
  };

  return (
    <div className="px-6 pb-4 flex-shrink-0">
      <div className="max-w-[1200px] mx-auto">
        {/* 推荐问题 */}
        {input === '' && !isTyping && (
          <div className="flex flex-wrap gap-2 mb-3">
            {RECOMMENDED_QUESTIONS.map((q) => (
              <button
                key={q.id}
                onClick={() => {
                  setInput(q.question);
                  textareaRef.current?.focus();
                }}
                className="recommend-tag"
              >
                {q.question}
              </button>
            ))}
          </div>
        )}

        {/* 输入框 */}
        <div className="relative flex items-end">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isTyping}
            placeholder="输入您的问题，Enter 发送，Shift+Enter 换行"
            rows={1}
            className="w-full min-h-[48px] max-h-[150px] px-4 py-3 pr-12 rounded-xl border border-border dark:border-border-dark bg-card dark:bg-card-dark text-text-main dark:text-text-mainDark placeholder:text-text-muted dark:placeholder:text-text-mutedDark resize-none focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/30 disabled:opacity-50 disabled:cursor-not-allowed text-[15px] leading-relaxed transition-colors"
          />
          {isTyping ? (
            <button
              onClick={handleStop}
              className="absolute right-2 bottom-2 w-8 h-8 flex items-center justify-center rounded-lg bg-red-500 text-white hover:bg-red-600 transition-all"
              title="停止生成"
            >
              <Square size={14} fill="currentColor" />
            </button>
          ) : (
            <button
              onClick={handleSend}
              disabled={!input.trim()}
              className="absolute right-2 bottom-2 w-8 h-8 flex items-center justify-center rounded-lg bg-primary text-white hover:bg-primary-hover disabled:opacity-40 disabled:cursor-not-allowed transition-all"
            >
              <ArrowUp size={18} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
