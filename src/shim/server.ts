import http from "node:http";
import { once } from "node:events";
import { mapOpenAIRequestToAifchem, mapOpenAIRequestToChatAppMessageApi } from "../aifchem/mapping.js";
import { OpenAIChatCompletionRequest, ShimConfig } from "../aifchem/types.js";
import { extractEventObjectsFromChunkBuffer, extractTextFromAifchemEvent } from "./parser.js";
import { createOpenAIStreamState, makeChunk, makeCompletion } from "./openai.js";

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

async function resolveChatAppThreadId(
  openAIRequest: OpenAIChatCompletionRequest,
  config: ShimConfig,
): Promise<string | undefined> {
  if (config.upstream.protocol !== "chat_app_message_api") {
    return undefined;
  }

  const chatApp = config.upstream.chatApp;
  if (!chatApp) {
    throw new Error("Missing upstream.chatApp configuration");
  }

  const metadataThreadId = getMetadataString(openAIRequest, "thread_id");
  if (metadataThreadId) {
    return metadataThreadId;
  }

  const shouldCreateThread =
    chatApp.createThreadPerRequest === true || (!chatApp.threadId && Boolean(chatApp.conversationId));
  if (!shouldCreateThread) {
    return chatApp.threadId;
  }

  const conversationId = getMetadataString(openAIRequest, "conversation_id") ?? chatApp.conversationId;
  if (!conversationId) {
    throw new Error("Missing chat_app_message_api conversation id for thread creation");
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

async function forwardToUpstream(
  openAIRequest: OpenAIChatCompletionRequest,
  config: ShimConfig,
  res: http.ServerResponse,
): Promise<void> {
  const protocol = config.upstream.protocol ?? "generic_aifchem";
  const threadIdOverride = await resolveChatAppThreadId(openAIRequest, config);
  const upstreamRequest =
    protocol === "chat_app_message_api"
      ? mapOpenAIRequestToChatAppMessageApi(openAIRequest, config, { threadId: threadIdOverride })
      : mapOpenAIRequestToAifchem(openAIRequest, config);
  const state = createOpenAIStreamState(openAIRequest.model);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config.upstream.timeoutMs ?? 30_000);

  try {
    let upstreamResponse: Response;
    try {
      upstreamResponse = await fetch(config.upstream.url, {
        method: config.upstream.method ?? "POST",
        headers: {
          "Content-Type": "application/json",
          ...(config.upstream.headers ?? {}),
        },
        body: JSON.stringify(upstreamRequest),
        signal: controller.signal,
      });
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
      jsonResponse(res, 502, { error: { message: await parseErrorMessage(upstreamResponse) } });
      return;
    }

    if (!upstreamResponse.body) {
      jsonResponse(res, 502, { error: { message: "Upstream response has no body" } });
      return;
    }

    const wantsStream = openAIRequest.stream === true;
    let previousContent = "";
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
        const extracted = extractTextFromAifchemEvent(event);

        if (extracted.done) {
          continue;
        }

        let delta = extracted.delta;
        if (typeof extracted.cumulative === "string") {
          if (extracted.cumulative.startsWith(previousContent)) {
            delta = extracted.cumulative.slice(previousContent.length);
            previousContent = extracted.cumulative;
            finalContent = extracted.cumulative;
          } else {
            delta = extracted.cumulative;
            finalContent += extracted.cumulative;
            previousContent = finalContent;
          }
        } else if (typeof delta === "string") {
          finalContent += delta;
          previousContent = finalContent;
        }

        if (!delta) {
          continue;
        }

        if (wantsStream) {
          res.write(`data: ${makeChunk(state, { content: delta }, null)}\n\n`);
        }
      }
    }

    if (buffer.trim()) {
      const fallback = JSON.parse(buffer) as Record<string, unknown>;
      if (typeof fallback.content === "string") {
        finalContent = fallback.content;
      }
    }

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
  const server = http.createServer(async (req, res) => {
    try {
      if (req.method === "GET" && req.url === "/health") {
        jsonResponse(res, 200, { ok: true });
        return;
      }

      if (req.method === "POST" && req.url === "/v1/chat/completions") {
        const body = await readJsonBody<OpenAIChatCompletionRequest>(req);
        await forwardToUpstream(body, config, res);
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
