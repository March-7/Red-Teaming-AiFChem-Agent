import http from "node:http";
import { once } from "node:events";
import { mapOpenAIRequestToAifchem, mapOpenAIRequestToChatAppMessageApi } from "../aifchem/mapping.js";
import { AifchemStreamEvent, OpenAIChatCompletionRequest, ShimConfig } from "../aifchem/types.js";
import { DataFenceFilter, extractEventObjectsFromChunkBuffer, extractTextFromAifchemEvent, sanitizeAssistantContent } from "./parser.js";
import { createOpenAIStreamState, makeChunk, makeCompletion } from "./openai.js";

const DEFAULT_CHAT_APP_SESSION_TTL_MS = 30 * 60 * 1000;

interface ChatAppSessionState {
  conversationId?: string;
  threadId?: string;
  workflowId?: string;
  parentMessageId?: string;
  updatedAt: number;
}

interface ResolvedChatAppContext {
  sessionId?: string;
  conversationId?: string;
  threadId?: string;
  workflowId?: string;
  parentMessageId?: string;
}

interface OutputFilterConfig {
  stripUiPayloads: boolean;
  stripDisclaimers: boolean;
}

function jsonResponse(res: http.ServerResponse, statusCode: number, body: unknown): void {
  res.writeHead(statusCode, { "Content-Type": "application/json; charset=utf-8" });
  res.end(`${JSON.stringify(body)}\n`);
}

async function readJsonBody<T>(req: http.IncomingMessage): Promise<T> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8")) as T;
}

function getMetadataString(request: OpenAIChatCompletionRequest, key: string): string | undefined {
  const value = request.metadata?.[key];
  return typeof value === "string" && value ? value : undefined;
}

function getSessionId(request: OpenAIChatCompletionRequest): string | undefined {
  return (
    getMetadataString(request, "session_id") ??
    getMetadataString(request, "agent_session_id") ??
    getMetadataString(request, "agent_id")
  );
}

function getChatAppSessionTtlMs(config: ShimConfig): number {
  return config.upstream.chatApp?.sessionTtlMs ?? DEFAULT_CHAT_APP_SESSION_TTL_MS;
}

function getOutputFilterConfig(config: ShimConfig): OutputFilterConfig {
  return {
    stripUiPayloads: config.upstream.chatApp?.filterUiPayloads !== false,
    stripDisclaimers: config.upstream.chatApp?.filterDisclaimers !== false,
  };
}

function readChatAppSession(
  sessions: Map<string, ChatAppSessionState>,
  sessionId: string | undefined,
  ttlMs: number,
): ChatAppSessionState | undefined {
  if (!sessionId) {
    return undefined;
  }

  const session = sessions.get(sessionId);
  if (!session) {
    return undefined;
  }

  if (Date.now() - session.updatedAt > ttlMs) {
    sessions.delete(sessionId);
    return undefined;
  }

  return session;
}

function writeChatAppSession(
  sessions: Map<string, ChatAppSessionState>,
  sessionId: string | undefined,
  state: ResolvedChatAppContext,
): void {
  if (!sessionId || !state.threadId) {
    return;
  }

  sessions.set(sessionId, {
    conversationId: state.conversationId,
    threadId: state.threadId,
    workflowId: state.workflowId,
    parentMessageId: state.parentMessageId,
    updatedAt: Date.now(),
  });
}

function updateChatAppContextFromEvent(context: ResolvedChatAppContext | undefined, event: AifchemStreamEvent): void {
  if (!context) {
    return;
  }

  if (typeof event.conversation_id === "string" && event.conversation_id) {
    context.conversationId = event.conversation_id;
  }
  if (typeof event.thread_id === "string" && event.thread_id) {
    context.threadId = event.thread_id;
  }
  if (typeof event.workflow_id === "string" && event.workflow_id) {
    context.workflowId = event.workflow_id;
  }
  if (typeof event.message_id === "string" && event.message_id) {
    context.parentMessageId = event.message_id;
  }
}

function shouldRetryWithFreshThread(message: string): boolean {
  return /unfinished messages in the current thread/i.test(message);
}

async function parseErrorMessage(response: Response): Promise<string> {
  let message = `Upstream request failed with ${response.status}`;

  try {
    const errorJson = (await response.json()) as Record<string, unknown>;
    if (typeof errorJson.message === "string") {
      return errorJson.message;
    }
    if (typeof errorJson.msg === "string") {
      return errorJson.msg;
    }
    if (errorJson.data && typeof errorJson.data === "object" && typeof (errorJson.data as Record<string, unknown>).message === "string") {
      return String((errorJson.data as Record<string, unknown>).message);
    }
  } catch {
    // Ignore malformed upstream error bodies.
  }

  return message;
}

async function createChatAppThread(conversationId: string, config: ShimConfig): Promise<string> {
  const chatApp = config.upstream.chatApp;
  if (!chatApp) {
    throw new Error("Missing upstream.chatApp configuration");
  }

  const threadCreateUrl = chatApp.threadCreateUrl ?? new URL("/api/thread/create", config.upstream.url).toString();
  const response = await fetch(threadCreateUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(config.upstream.headers ?? {}),
    },
    body: JSON.stringify({ conversation_id: conversationId }),
  });

  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }

  const payload = (await response.json()) as Record<string, unknown>;
  const data = payload.data as Record<string, unknown> | undefined;
  const threadId = typeof data?.id === "string" ? data.id : undefined;
  if (!threadId) {
    throw new Error("Thread creation response did not include data.id");
  }

  return threadId;
}

async function resolveChatAppContext(
  openAIRequest: OpenAIChatCompletionRequest,
  config: ShimConfig,
  sessions: Map<string, ChatAppSessionState>,
): Promise<ResolvedChatAppContext | undefined> {
  if (config.upstream.protocol !== "chat_app_message_api") {
    return undefined;
  }

  const chatApp = config.upstream.chatApp;
  if (!chatApp) {
    throw new Error("Missing upstream.chatApp configuration");
  }

  const sessionId = getSessionId(openAIRequest);
  const session = readChatAppSession(sessions, sessionId, getChatAppSessionTtlMs(config));
  const reuseThreadWithinSession = chatApp.reuseThreadWithinSession !== false;

  const context: ResolvedChatAppContext = {
    sessionId,
    conversationId:
      getMetadataString(openAIRequest, "conversation_id") ?? session?.conversationId ?? chatApp.conversationId,
    threadId:
      getMetadataString(openAIRequest, "thread_id") ??
      (reuseThreadWithinSession ? session?.threadId : undefined) ??
      chatApp.threadId,
    workflowId: getMetadataString(openAIRequest, "workflow_id") ?? session?.workflowId ?? chatApp.workflowId,
    parentMessageId:
      getMetadataString(openAIRequest, "parent_message_id") ??
      (reuseThreadWithinSession ? session?.parentMessageId : undefined) ??
      chatApp.parentMessageId ??
      "",
  };

  if (context.threadId) {
    return context;
  }

  const shouldCreateThread =
    chatApp.createThreadPerRequest === true ||
    (Boolean(context.sessionId) && reuseThreadWithinSession) ||
    Boolean(context.conversationId);

  if (!shouldCreateThread) {
    return context;
  }

  if (!context.conversationId) {
    throw new Error("Missing chat_app_message_api conversation id for thread creation");
  }

  context.threadId = await createChatAppThread(context.conversationId, config);
  context.parentMessageId = getMetadataString(openAIRequest, "parent_message_id") ?? chatApp.parentMessageId ?? "";
  return context;
}

async function createFreshChatAppContext(
  openAIRequest: OpenAIChatCompletionRequest,
  config: ShimConfig,
  previousContext: ResolvedChatAppContext | undefined,
): Promise<ResolvedChatAppContext> {
  const chatApp = config.upstream.chatApp;
  if (!chatApp) {
    throw new Error("Missing upstream.chatApp configuration");
  }

  const conversationId =
    previousContext?.conversationId ??
    getMetadataString(openAIRequest, "conversation_id") ??
    chatApp.conversationId;

  if (!conversationId) {
    throw new Error("Missing chat_app_message_api conversation id for thread creation");
  }

  return {
    sessionId: previousContext?.sessionId ?? getSessionId(openAIRequest),
    conversationId,
    threadId: await createChatAppThread(conversationId, config),
    workflowId: previousContext?.workflowId ?? getMetadataString(openAIRequest, "workflow_id") ?? chatApp.workflowId,
    parentMessageId: "",
  };
}

async function fetchUpstream(
  config: ShimConfig,
  upstreamRequest: Record<string, unknown>,
  signal: AbortSignal,
): Promise<Response> {
  return fetch(config.upstream.url, {
    method: config.upstream.method ?? "POST",
    headers: {
      "Content-Type": "application/json",
      ...(config.upstream.headers ?? {}),
    },
    body: JSON.stringify(upstreamRequest),
    signal,
  });
}

async function forwardToUpstream(
  openAIRequest: OpenAIChatCompletionRequest,
  config: ShimConfig,
  res: http.ServerResponse,
  sessions: Map<string, ChatAppSessionState>,
): Promise<void> {
  const protocol = config.upstream.protocol ?? "generic_aifchem";
  const outputFilterConfig = getOutputFilterConfig(config);
  let chatAppContext = await resolveChatAppContext(openAIRequest, config, sessions);
  let upstreamRequest =
    protocol === "chat_app_message_api"
      ? mapOpenAIRequestToChatAppMessageApi(openAIRequest, config, {
          threadId: chatAppContext?.threadId,
          workflowId: chatAppContext?.workflowId,
          parentMessageId: chatAppContext?.parentMessageId,
        })
      : mapOpenAIRequestToAifchem(openAIRequest, config);
  const state = createOpenAIStreamState(openAIRequest.model);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config.upstream.timeoutMs ?? 30_000);

  try {
    let upstreamResponse: Response;
    try {
      upstreamResponse = await fetchUpstream(config, upstreamRequest, controller.signal);
    } catch (error) {
      const details = error instanceof Error ? error.message : String(error);
      jsonResponse(res, 502, {
        error: {
          message: `Failed to reach upstream ${config.upstream.url}: ${details}`,
        },
      });
      return;
    }

    if (!upstreamResponse.ok) {
      const upstreamErrorMessage = await parseErrorMessage(upstreamResponse);

      if (protocol === "chat_app_message_api" && shouldRetryWithFreshThread(upstreamErrorMessage)) {
        try {
          chatAppContext = await createFreshChatAppContext(openAIRequest, config, chatAppContext);
          upstreamRequest = mapOpenAIRequestToChatAppMessageApi(openAIRequest, config, {
            threadId: chatAppContext.threadId,
            workflowId: chatAppContext.workflowId,
            parentMessageId: "",
          });
          upstreamResponse = await fetchUpstream(config, upstreamRequest, controller.signal);
        } catch (error) {
          jsonResponse(res, 502, {
            error: {
              message: error instanceof Error ? error.message : String(error),
            },
          });
          return;
        }

        if (!upstreamResponse.ok) {
          jsonResponse(res, 502, { error: { message: await parseErrorMessage(upstreamResponse) } });
          return;
        }
      } else {
        jsonResponse(res, 502, { error: { message: upstreamErrorMessage } });
        return;
      }
    }

    if (!upstreamResponse.body) {
      jsonResponse(res, 502, { error: { message: "Upstream response has no body" } });
      return;
    }

    const wantsStream = openAIRequest.stream === true;
    const contentFilter = outputFilterConfig.stripUiPayloads ? new DataFenceFilter() : undefined;
    let rawContent = "";
    let finalContent = "";

    if (wantsStream) {
      res.writeHead(200, {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
      });
      res.write(`data: ${makeChunk(state, { role: "assistant" }, null)}\n\n`);
    }

    const reader = upstreamResponse.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const parsed = extractEventObjectsFromChunkBuffer(buffer);
      buffer = parsed.rest;

      for (const event of parsed.events) {
        updateChatAppContextFromEvent(chatAppContext, event);
        const extracted = extractTextFromAifchemEvent(event);

        if (extracted.done) {
          continue;
        }

        let rawDelta = extracted.delta;
        if (typeof extracted.cumulative === "string") {
          if (extracted.cumulative.startsWith(rawContent)) {
            rawDelta = extracted.cumulative.slice(rawContent.length);
            rawContent = extracted.cumulative;
          } else {
            rawDelta = extracted.cumulative;
            rawContent += extracted.cumulative;
          }
        } else if (typeof rawDelta === "string") {
          rawContent += rawDelta;
        }

        if (!rawDelta) {
          continue;
        }

        const delta = contentFilter ? contentFilter.push(rawDelta) : rawDelta;
        if (!delta) {
          continue;
        }

        finalContent += delta;

        if (wantsStream) {
          res.write(`data: ${makeChunk(state, { content: delta }, null)}\n\n`);
        }
      }
    }

    if (buffer.trim()) {
      const fallback = JSON.parse(buffer) as AifchemStreamEvent;
      updateChatAppContextFromEvent(chatAppContext, fallback);
      if (typeof fallback.content === "string") {
        finalContent = sanitizeAssistantContent(fallback.content, outputFilterConfig);
      }
    }

    const trailingContent = contentFilter?.flush() ?? "";
    if (trailingContent) {
      finalContent += trailingContent;
      if (wantsStream) {
        res.write(`data: ${makeChunk(state, { content: trailingContent }, null)}\n\n`);
      }
    }

    if (protocol === "chat_app_message_api") {
      writeChatAppSession(sessions, chatAppContext?.sessionId, chatAppContext ?? {});
    }

    finalContent = sanitizeAssistantContent(finalContent, {
      stripUiPayloads: false,
      stripDisclaimers: outputFilterConfig.stripDisclaimers,
    });

    if (wantsStream) {
      res.write(`data: ${makeChunk(state, {}, "stop")}\n\n`);
      res.write("data: [DONE]\n\n");
      res.end();
      return;
    }

    jsonResponse(res, 200, makeCompletion(state, finalContent));
  } finally {
    clearTimeout(timeout);
  }
}

export async function createShimServer(config: ShimConfig): Promise<http.Server> {
  const chatAppSessions = new Map<string, ChatAppSessionState>();
  const server = http.createServer(async (req, res) => {
    try {
      if (req.method === "GET" && req.url === "/health") {
        jsonResponse(res, 200, { ok: true });
        return;
      }

      if (req.method === "POST" && req.url === "/v1/chat/completions") {
        const body = await readJsonBody<OpenAIChatCompletionRequest>(req);
        await forwardToUpstream(body, config, res, chatAppSessions);
        return;
      }

      jsonResponse(res, 404, { error: { message: "Not found" } });
    } catch (error) {
      jsonResponse(res, 500, {
        error: {
          message: error instanceof Error ? error.message : String(error),
        },
      });
    }
  });

  server.listen(config.listen.port, config.listen.host);
  await once(server, "listening");
  return server;
}
