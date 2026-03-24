export interface AifchemMessage {
  role: "system" | "user" | "assistant" | "tool" | string;
  content: string;
}

export interface AifchemRequest {
  message_id: string;
  messages: AifchemMessage[];
  context_link?: string;
  response_mode: "streaming" | "blocking";
  [key: string]: unknown;
}

export interface AifchemStreamEvent {
  content?: string;
  delta?: string;
  text?: string;
  done?: boolean;
  error?: string;
  [key: string]: unknown;
}

export interface OpenAIChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | Array<{ type: string; text?: string }>;
}

export interface OpenAIChatCompletionRequest {
  model: string;
  messages: OpenAIChatMessage[];
  stream?: boolean;
  user?: string;
  metadata?: Record<string, unknown>;
}

export interface ShimConfig {
  listen: {
    host: string;
    port: number;
  };
  upstream: {
    protocol?: "generic_aifchem" | "chat_app_message_api";
    url: string;
    method?: string;
    headers?: Record<string, string>;
    timeoutMs?: number;
    contextLink?: string;
    messageRoleMode?: "preserve" | "user_only_concat";
    extraBody?: Record<string, unknown>;
    chatApp?: {
      threadId?: string;
      conversationId?: string;
      threadCreateUrl?: string;
      createThreadPerRequest?: boolean;
      workflowId: string;
      parentMessageId?: string;
      inputs?: Record<string, unknown>;
    };
  };
}
