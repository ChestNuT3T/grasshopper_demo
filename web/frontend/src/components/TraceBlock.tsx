/**
 * 检索 Trace 组件：扁平化检索结果展示
 */

import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

interface TraceBlockProps {
  trace: Record<string, unknown>;
}

export function TraceBlock({ trace }: TraceBlockProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const renderTrace = (obj: Record<string, unknown>, indent = 0): JSX.Element[] => {
    const elements: JSX.Element[] = [];
    for (const [key, value] of Object.entries(obj)) {
      elements.push(
        <div key={key} style={{ marginLeft: `${indent * 14}px` }} className="py-0.5">
          <span className="text-text-sub dark:text-text-subDark font-medium text-sm">{key}:</span>
          {typeof value === 'object' && value !== null ? (
            <div className="mt-0.5">
              {renderTrace(value as Record<string, unknown>, indent + 1)}
            </div>
          ) : (
            <span className="ml-2 text-sm text-text-main dark:text-text-mainDark">
              {String(value)}
            </span>
          )}
        </div>
      );
    }
    return elements;
  };

  return (
    <div className="trace-block mt-2">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-1.5 text-sm text-text-sub dark:text-text-subDark hover:text-text-main dark:hover:text-text-mainDark transition-colors w-full"
      >
        {isExpanded ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
        <span>检索 Trace</span>
      </button>
      {isExpanded && (
        <div className="mt-2 text-sm">{renderTrace(trace)}</div>
      )}
    </div>
  );
}