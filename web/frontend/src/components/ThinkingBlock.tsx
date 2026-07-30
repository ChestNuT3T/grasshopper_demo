/**
 * 思考过程块组件：DeepSeek 风格思考展示
 * 
 * 功能：
 *   - 思考中自动展开，实时流式展示思考内容
 *   - 思考结束后自动折叠
 *   - 思考中显示动态状态
 *   - 思考结束显示用时
 *   - 用户可手动点击展开/折叠
 */

import { useState, useEffect, useRef } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

interface ThinkingBlockProps {
  content: string;
  isThinking?: boolean;
  thinkingTime?: number;
}

export function ThinkingBlock({ content, isThinking = false, thinkingTime = 0 }: ThinkingBlockProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const wasThinkingRef = useRef(false);

  useEffect(() => {
    if (isThinking) {
      wasThinkingRef.current = true;
      setIsExpanded(true);
    } else if (wasThinkingRef.current) {
      wasThinkingRef.current = false;
      setIsExpanded(false);
    }
  }, [isThinking]);

  const formatTime = (seconds: number) => {
    if (seconds < 60) {
      return `${Math.round(seconds)} 秒`;
    }
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${mins} 分 ${secs} 秒`;
  };

  return (
    <div className="mb-2">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-1.5 text-sm text-text-sub dark:text-text-subDark hover:text-text-main dark:hover:text-text-mainDark transition-colors w-full"
      >
        {isExpanded ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
        <span>
          {isThinking ? '思考中...' : `已深度思考（用时 ${formatTime(thinkingTime)}）`}
        </span>
      </button>
      {isExpanded && content && (
        <div className="thinking-block mt-2">
          <div className="thinking-content">{content}</div>
        </div>
      )}
    </div>
  );
}
