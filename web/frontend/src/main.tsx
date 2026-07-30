/**
 * 应用入口文件：渲染 React 应用到 DOM
 * 
 * 配置：
 *   - React 18 新的渲染 API（createRoot）
 *   - 全局样式导入（index.css）
 *   - Tailwind CSS 主题配置
 */

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App';

/**
 * 创建 React 根节点并渲染应用
 */
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
