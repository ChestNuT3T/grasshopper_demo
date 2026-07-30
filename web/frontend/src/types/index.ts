/**
 * 类型定义模块：定义项目中使用的所有 TypeScript 类型
 */

/**
 * 消息接口
 * 
 * @interface Message
 * @property id - 消息唯一标识
 * @property role - 消息角色（user/assistant）
 * @property content - 消息内容（Markdown 格式）
 * @property reasoning - 思考过程（可选）
 * @property trace - 检索 Trace 数据（可选）
 * @property createdAt - 创建时间戳
 */
export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  reasoning?: string;
  trace?: Record<string, unknown>;
  thinkingTime?: number;
  createdAt: number;
}

/**
 * 会话接口
 * 
 * @interface Session
 * @property id - 会话唯一标识
 * @property name - 会话名称
 * @property messageCount - 消息数量
 * @property lastModified - 最后修改时间戳
 */
export interface Session {
  id: string;
  name: string;
  messageCount: number;
  lastModified: number;
}

/**
 * 流式数据块接口
 * 
 * @interface StreamChunk
 * @property type - 块类型（reasoning/content/end）
 * @property content - 内容块（可选）
 * @property reasoning - 思考过程块（可选）
 * @property trace - Trace 数据（可选）
 */
export interface StreamChunk {
  type: 'reasoning' | 'content' | 'end' | 'error';
  content?: string;
  reasoning?: string;
  trace?: Record<string, unknown>;
}

/**
 * 聊天请求接口
 * 
 * @interface ChatRequest
 * @property user_input - 用户输入
 * @property session_id - 会话 ID
 * @property rag_enabled - 是否启用 RAG 检索
 */
export interface ChatRequest {
  user_input: string;
  session_id: string;
  rag_enabled: boolean;
}

/**
 * 会话响应接口
 * 
 * @interface ChatResponse
 * @property messages - 消息列表
 */
export interface ChatResponse {
  messages: Message[];
}

/**
 * 会话列表响应接口
 * 
 * @interface SessionsResponse
 * @property sessions - 会话列表
 */
export interface SessionsResponse {
  sessions: Session[];
}

/**
 * 推荐问题接口
 * 
 * @interface RecommendedQuestion
 * @property id - 问题唯一标识
 * @property question - 问题内容
 */
export interface RecommendedQuestion {
  id: string;
  question: string;
}

/**
 * 聊天状态接口（全局状态）
 * 
 * @interface ChatState
 * @property sessions - 会话列表
 * @property currentSessionId - 当前会话 ID
 * @property messages - 当前会话的消息列表
 * @property isTyping - 是否正在生成回答
 * @property ragEnabled - RAG 检索是否启用
 * @property theme - 主题模式（light/dark/system）
 * @property sidebarCollapsed - 侧边栏是否折叠
 * @property error - 错误信息
 */
export interface ChatState {
  sessions: Session[];
  currentSessionId: string;
  messages: Message[];
  isTyping: boolean;
  ragEnabled: boolean;
  theme: 'light' | 'dark' | 'system';
  sidebarCollapsed: boolean;
  error: string | null;
}

/**
 * 默认推荐问题列表
 */
export const RECOMMENDED_QUESTIONS: RecommendedQuestion[] = [
  { id: '1', question: 'Elefront 怎么烘焙属性？' },
  { id: '2', question: 'Isotrim 报错怎么办？' },
  { id: '3', question: '如何生成双曲面分格？' },
];
