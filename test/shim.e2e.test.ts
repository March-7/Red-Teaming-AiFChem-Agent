import test from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { once } from "node:events";
import { createShimServer } from "../src/shim/server.js";
import { ShimConfig } from "../src/aifchem/types.js";

async function listen(server: http.Server): Promise<number> {
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("Failed to get listening address");
  }
  return address.port;
}

async function createMockUpstream(capture: { body?: unknown }): Promise<{ server: http.Server; port: number }> {
  const server = http.createServer(async (req, res) => {
    if (req.method !== "POST") {
      res.writeHead(405).end();
      return;
    }

    const chunks: Buffer[] = [];
    for await (const chunk of req) {
      chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    }
    capture.body = JSON.parse(Buffer.concat(chunks).toString("utf8"));

    res.writeHead(200, {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });
    res.write('data: {"content":"Hello"}\n\n');
    res.write('data: {"content":"Hello world"}\n\n');
    res.write('data: {"done":true}\n\n');
    res.end();
  });

  const port = await listen(server);
  return { server, port };
}

async function createMockChatAppUpstream(capture: { threadBodies: any[]; chatBodies: any[] }): Promise<{ server: http.Server; port: number }> {
  const server = http.createServer(async (req, res) => {
    const chunks: Buffer[] = [];
    for await (const chunk of req) {
      chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    }
    const body = chunks.length > 0 ? JSON.parse(Buffer.concat(chunks).toString("utf8")) : undefined;

    if (req.method === "POST" && req.url === "/api/thread/create") {
      capture.threadBodies.push(body);
      res.writeHead(201, { "Content-Type": "application/json; charset=utf-8" });
      res.end(
        JSON.stringify({
          data: {
            id: "generated-thread-id",
          },
          code: 200,
          msg: "",
        }),
      );
      return;
    }

    if (req.method === "POST" && req.url === "/api/message/chat") {
      capture.chatBodies.push(body);
      res.writeHead(200, {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      });
      res.write(
        `data: ${JSON.stringify({
          conversation_id: "conversation-123",
          thread_id: body.thread_id,
          workflow_id: body.workflow_id,
          message_id: "assistant-message-1",
          content: "Hello",
        })}\n\n`,
      );
      res.write(
        `data: ${JSON.stringify({
          conversation_id: "conversation-123",
          thread_id: body.thread_id,
          workflow_id: body.workflow_id,
          message_id: "assistant-message-1",
          content: " world",
        })}\n\n`,
      );
      res.write('data: {"done":true}\n\n');
      res.end();
      return;
    }

    res.writeHead(404).end();
  });

  const port = await listen(server);
  return { server, port };
}

async function createStatefulMockChatAppUpstream(
  capture: { threadBodies: any[]; chatBodies: any[] },
  options: {
    onThreadCreate?: (count: number, body: any) => { threadId: string };
    onChat?: (
      count: number,
      body: any,
    ) =>
      | {
          statusCode?: number;
          errorBody?: Record<string, unknown>;
        }
      | {
          events: Array<Record<string, unknown>>;
        };
  } = {},
): Promise<{ server: http.Server; port: number }> {
  const server = http.createServer(async (req, res) => {
    const chunks: Buffer[] = [];
    for await (const chunk of req) {
      chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    }
    const body = chunks.length > 0 ? JSON.parse(Buffer.concat(chunks).toString("utf8")) : undefined;

    if (req.method === "POST" && req.url === "/api/thread/create") {
      capture.threadBodies.push(body);
      const result = options.onThreadCreate?.(capture.threadBodies.length, body) ?? {
        threadId: `generated-thread-${capture.threadBodies.length}`,
      };
      res.writeHead(201, { "Content-Type": "application/json; charset=utf-8" });
      res.end(
        JSON.stringify({
          data: {
            id: result.threadId,
          },
          code: 200,
          msg: "",
        }),
      );
      return;
    }

    if (req.method === "POST" && req.url === "/api/message/chat") {
      capture.chatBodies.push(body);
      const result = options.onChat?.(capture.chatBodies.length, body) ?? {
        events: [
          {
            conversation_id: "conversation-123",
            thread_id: body.thread_id,
            workflow_id: body.workflow_id,
            message_id: `assistant-message-${capture.chatBodies.length}`,
            content: "Hello world",
          },
          { done: true },
        ],
      };

      if ("statusCode" in result && result.statusCode && result.statusCode >= 400) {
        const payload = result.errorBody ?? { message: "Upstream request failed" };
        const encoded = JSON.stringify(payload);
        res.writeHead(result.statusCode, { "Content-Type": "application/json; charset=utf-8" });
        res.end(encoded);
        return;
      }

      res.writeHead(200, {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      });
      for (const event of result.events) {
        res.write(`data: ${JSON.stringify(event)}\n\n`);
      }
      res.end();
      return;
    }

    res.writeHead(404).end();
  });

  const port = await listen(server);
  return { server, port };
}

test("shim converts OpenAI non-stream request to AIFChem and returns chat completion", async () => {
  const capture: { body?: any } = {};
  const upstream = await createMockUpstream(capture);

  const config: ShimConfig = {
    listen: { host: "127.0.0.1", port: 0 },
    upstream: {
      url: `http://127.0.0.1:${upstream.port}/agent`,
      method: "POST",
      headers: {
        Authorization: "Bearer test-token",
      },
      messageRoleMode: "preserve",
    },
  };

  const shim = await createShimServer(config);
  const shimPort = typeof shim.address() === "object" && shim.address() ? shim.address()!.port : 0;

  const response = await fetch(`http://127.0.0.1:${shimPort}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: "Bearer sk-local-test",
    },
    body: JSON.stringify({
      model: "aifchem-agent",
      messages: [
        { role: "system", content: "You are safe." },
        { role: "user", content: "hello" },
      ],
      stream: false,
    }),
  });

  assert.equal(response.status, 200);
  const json = (await response.json()) as any;
  assert.equal(json.object, "chat.completion");
  assert.equal(json.choices[0].message.content, "Hello world");
  assert.equal(capture.body.response_mode, "streaming");
  assert.equal(capture.body.messages[0].role, "system");
  assert.equal(capture.body.messages[1].role, "user");

  await new Promise<void>((resolve, reject) => shim.close((error) => (error ? reject(error) : resolve())));
  await new Promise<void>((resolve, reject) => upstream.server.close((error) => (error ? reject(error) : resolve())));
});

test("shim converts upstream cumulative content stream into OpenAI SSE delta stream", async () => {
  const capture: { body?: any } = {};
  const upstream = await createMockUpstream(capture);

  const config: ShimConfig = {
    listen: { host: "127.0.0.1", port: 0 },
    upstream: {
      url: `http://127.0.0.1:${upstream.port}/agent`,
      method: "POST",
      messageRoleMode: "user_only_concat",
      contextLink: "context://test",
    },
  };

  const shim = await createShimServer(config);
  const shimPort = typeof shim.address() === "object" && shim.address() ? shim.address()!.port : 0;

  const response = await fetch(`http://127.0.0.1:${shimPort}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "aifchem-agent",
      messages: [
        { role: "system", content: "safe" },
        { role: "assistant", content: "previous" },
        { role: "user", content: "hello" },
      ],
      stream: true,
    }),
  });

  assert.equal(response.status, 200);
  const text = await response.text();
  assert.match(text, /"role":"assistant"/);
  assert.match(text, /"content":"Hello"/);
  assert.match(text, /"content":" world"/);
  assert.match(text, /\[DONE\]/);
  assert.equal(capture.body.context_link, "context://test");
  assert.equal(capture.body.messages[0].role, "user");
  assert.match(capture.body.messages[0].content, /^SYSTEM:/);

  await new Promise<void>((resolve, reject) => shim.close((error) => (error ? reject(error) : resolve())));
  await new Promise<void>((resolve, reject) => upstream.server.close((error) => (error ? reject(error) : resolve())));
});

test("shim creates a fresh chat-app thread before forwarding the chat request", async () => {
  const capture = { threadBodies: [] as any[], chatBodies: [] as any[] };
  const upstream = await createMockChatAppUpstream(capture);

  const config: ShimConfig = {
    listen: { host: "127.0.0.1", port: 0 },
    upstream: {
      protocol: "chat_app_message_api",
      url: `http://127.0.0.1:${upstream.port}/api/message/chat`,
      method: "POST",
      headers: {
        Authorization: "Bearer test-token",
      },
      chatApp: {
        conversationId: "conversation-123",
        workflowId: "workflow-456",
        createThreadPerRequest: true,
      },
    },
  };

  const shim = await createShimServer(config);
  const shimPort = typeof shim.address() === "object" && shim.address() ? shim.address()!.port : 0;

  const response = await fetch(`http://127.0.0.1:${shimPort}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: "Bearer sk-local-test",
    },
    body: JSON.stringify({
      model: "aifchem-agent",
      messages: [{ role: "user", content: "hello" }],
      stream: false,
    }),
  });

  assert.equal(response.status, 200);
  const json = (await response.json()) as any;
  assert.equal(json.choices[0].message.content, "Hello world");
  assert.deepEqual(capture.threadBodies, [{ conversation_id: "conversation-123" }]);
  assert.equal(capture.chatBodies[0].thread_id, "generated-thread-id");
  assert.equal(capture.chatBodies[0].workflow_id, "workflow-456");
  assert.equal(capture.chatBodies[0].query, "hello");

  await new Promise<void>((resolve, reject) => shim.close((error) => (error ? reject(error) : resolve())));
  await new Promise<void>((resolve, reject) => upstream.server.close((error) => (error ? reject(error) : resolve())));
});

test("shim serializes full OpenAI conversation history into chat-app query", async () => {
  const capture = { threadBodies: [] as any[], chatBodies: [] as any[] };
  const upstream = await createMockChatAppUpstream(capture);

  const config: ShimConfig = {
    listen: { host: "127.0.0.1", port: 0 },
    upstream: {
      protocol: "chat_app_message_api",
      url: `http://127.0.0.1:${upstream.port}/api/message/chat`,
      method: "POST",
      chatApp: {
        conversationId: "conversation-123",
        workflowId: "workflow-456",
        createThreadPerRequest: true,
      },
    },
  };

  const shim = await createShimServer(config);
  const shimPort = typeof shim.address() === "object" && shim.address() ? shim.address()!.port : 0;

  const response = await fetch(`http://127.0.0.1:${shimPort}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "aifchem-agent",
      messages: [
        { role: "system", content: "You are a careful chemistry assistant." },
        { role: "user", content: "I mixed reagent A." },
        { role: "assistant", content: "What solvent did you use?" },
        { role: "tool", content: "solvent=water" },
        { role: "user", content: "What should I do next?" },
      ],
      stream: false,
    }),
  });

  assert.equal(response.status, 200);
  const query = capture.chatBodies[0].query as string;
  assert.match(query, /full conversation transcript so far/i);
  assert.match(query, /SYSTEM:\nYou are a careful chemistry assistant\./);
  assert.match(query, /ASSISTANT:\nWhat solvent did you use\?/);
  assert.match(query, /TOOL:\nsolvent=water/);
  assert.match(query, /USER:\nWhat should I do next\?/);

  await new Promise<void>((resolve, reject) => shim.close((error) => (error ? reject(error) : resolve())));
  await new Promise<void>((resolve, reject) => upstream.server.close((error) => (error ? reject(error) : resolve())));
});

test("shim reuses chat-app thread and parent message across turns for the same session", async () => {
  const capture = { threadBodies: [] as any[], chatBodies: [] as any[] };
  const upstream = await createStatefulMockChatAppUpstream(capture, {
    onChat: (count, body) => ({
      events: [
        {
          conversation_id: "conversation-123",
          thread_id: body.thread_id,
          workflow_id: body.workflow_id,
          message_id: `assistant-message-${count}`,
          content: count === 1 ? "Turn one" : "Turn two",
        },
        { done: true },
      ],
    }),
  });

  const config: ShimConfig = {
    listen: { host: "127.0.0.1", port: 0 },
    upstream: {
      protocol: "chat_app_message_api",
      url: `http://127.0.0.1:${upstream.port}/api/message/chat`,
      method: "POST",
      chatApp: {
        conversationId: "conversation-123",
        workflowId: "workflow-456",
        createThreadPerRequest: true,
      },
    },
  };

  const shim = await createShimServer(config);
  const shimPort = typeof shim.address() === "object" && shim.address() ? shim.address()!.port : 0;

  const firstResponse = await fetch(`http://127.0.0.1:${shimPort}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "aifchem-agent",
      metadata: {
        session_id: "agent-session-1",
      },
      messages: [{ role: "user", content: "hello" }],
      stream: false,
    }),
  });

  const secondResponse = await fetch(`http://127.0.0.1:${shimPort}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "aifchem-agent",
      metadata: {
        session_id: "agent-session-1",
      },
      messages: [{ role: "user", content: "follow up" }],
      stream: false,
    }),
  });

  assert.equal(firstResponse.status, 200);
  assert.equal(secondResponse.status, 200);
  assert.equal(capture.threadBodies.length, 1);
  assert.equal(capture.chatBodies.length, 2);
  assert.equal(capture.chatBodies[0].thread_id, "generated-thread-1");
  assert.equal(capture.chatBodies[0].parent_message_id, "");
  assert.equal(capture.chatBodies[1].thread_id, "generated-thread-1");
  assert.equal(capture.chatBodies[1].parent_message_id, "assistant-message-1");

  await new Promise<void>((resolve, reject) => shim.close((error) => (error ? reject(error) : resolve())));
  await new Promise<void>((resolve, reject) => upstream.server.close((error) => (error ? reject(error) : resolve())));
});

test("shim falls back to a fresh thread when a reused chat-app thread is rejected", async () => {
  const capture = { threadBodies: [] as any[], chatBodies: [] as any[] };
  const upstream = await createStatefulMockChatAppUpstream(capture, {
    onChat: (count, body) => {
      if (count === 2) {
        return {
          statusCode: 409,
          errorBody: {
            message: "There are unfinished messages in the current thread",
          },
        };
      }

      return {
        events: [
          {
            conversation_id: "conversation-123",
            thread_id: body.thread_id,
            workflow_id: body.workflow_id,
            message_id: `assistant-message-${count}`,
            content: count === 1 ? "Turn one" : "Recovered turn two",
          },
          { done: true },
        ],
      };
    },
  });

  const config: ShimConfig = {
    listen: { host: "127.0.0.1", port: 0 },
    upstream: {
      protocol: "chat_app_message_api",
      url: `http://127.0.0.1:${upstream.port}/api/message/chat`,
      method: "POST",
      chatApp: {
        conversationId: "conversation-123",
        workflowId: "workflow-456",
        createThreadPerRequest: true,
      },
    },
  };

  const shim = await createShimServer(config);
  const shimPort = typeof shim.address() === "object" && shim.address() ? shim.address()!.port : 0;

  await fetch(`http://127.0.0.1:${shimPort}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "aifchem-agent",
      metadata: {
        session_id: "agent-session-2",
      },
      messages: [{ role: "user", content: "hello" }],
      stream: false,
    }),
  });

  const secondResponse = await fetch(`http://127.0.0.1:${shimPort}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "aifchem-agent",
      metadata: {
        session_id: "agent-session-2",
      },
      messages: [
        { role: "user", content: "hello" },
        { role: "assistant", content: "Turn one" },
        { role: "user", content: "follow up" },
      ],
      stream: false,
    }),
  });

  assert.equal(secondResponse.status, 200);
  const json = (await secondResponse.json()) as any;
  assert.equal(json.choices[0].message.content, "Recovered turn two");
  assert.equal(capture.threadBodies.length, 2);
  assert.equal(capture.chatBodies.length, 3);
  assert.equal(capture.chatBodies[1].thread_id, "generated-thread-1");
  assert.equal(capture.chatBodies[1].parent_message_id, "assistant-message-1");
  assert.equal(capture.chatBodies[2].thread_id, "generated-thread-2");
  assert.equal(capture.chatBodies[2].parent_message_id, "");
  assert.match(capture.chatBodies[2].query, /ASSISTANT:\nTurn one/);

  await new Promise<void>((resolve, reject) => shim.close((error) => (error ? reject(error) : resolve())));
  await new Promise<void>((resolve, reject) => upstream.server.close((error) => (error ? reject(error) : resolve())));
});

test("shim strips chat-app fenced data payloads from assistant output", async () => {
  const capture = { threadBodies: [] as any[], chatBodies: [] as any[] };
  const upstream = await createStatefulMockChatAppUpstream(capture, {
    onChat: (_count, body) => ({
      events: [
        {
          conversation_id: "conversation-123",
          thread_id: body.thread_id,
          workflow_id: body.workflow_id,
          message_id: "assistant-message-1",
          delta: "You said your name is Xiaojin.\n\n```da",
        },
        {
          conversation_id: "conversation-123",
          thread_id: body.thread_id,
          workflow_id: body.workflow_id,
          message_id: "assistant-message-1",
          delta:
            'ta {"data_type":"form","content":{"title":"Get In Touch","schema":{"fields":[]}}}',
        },
        {
          conversation_id: "conversation-123",
          thread_id: body.thread_id,
          workflow_id: body.workflow_id,
          message_id: "assistant-message-1",
          delta: "```",
        },
        { done: true },
      ],
    }),
  });

  const config: ShimConfig = {
    listen: { host: "127.0.0.1", port: 0 },
    upstream: {
      protocol: "chat_app_message_api",
      url: `http://127.0.0.1:${upstream.port}/api/message/chat`,
      method: "POST",
      chatApp: {
        conversationId: "conversation-123",
        workflowId: "workflow-456",
        createThreadPerRequest: true,
      },
    },
  };

  const shim = await createShimServer(config);
  const shimPort = typeof shim.address() === "object" && shim.address() ? shim.address()!.port : 0;

  const response = await fetch(`http://127.0.0.1:${shimPort}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "aifchem-agent",
      messages: [{ role: "user", content: "我是谁？" }],
      stream: false,
    }),
  });

  assert.equal(response.status, 200);
  const json = (await response.json()) as any;
  assert.equal(json.choices[0].message.content, "You said your name is Xiaojin.");

  await new Promise<void>((resolve, reject) => shim.close((error) => (error ? reject(error) : resolve())));
  await new Promise<void>((resolve, reject) => upstream.server.close((error) => (error ? reject(error) : resolve())));
});

test("shim strips fixed LLM disclaimer from assistant output", async () => {
  const capture = { threadBodies: [] as any[], chatBodies: [] as any[] };
  const upstream = await createStatefulMockChatAppUpstream(capture, {
    onChat: (_count, body) => ({
      events: [
        {
          conversation_id: "conversation-123",
          thread_id: body.thread_id,
          workflow_id: body.workflow_id,
          message_id: "assistant-message-1",
          content:
            "You said your name is Xiaojin.\n\nThis answer is generated by a Large Language Model, please verify the information independently.",
        },
        { done: true },
      ],
    }),
  });

  const config: ShimConfig = {
    listen: { host: "127.0.0.1", port: 0 },
    upstream: {
      protocol: "chat_app_message_api",
      url: `http://127.0.0.1:${upstream.port}/api/message/chat`,
      method: "POST",
      chatApp: {
        conversationId: "conversation-123",
        workflowId: "workflow-456",
        createThreadPerRequest: true,
      },
    },
  };

  const shim = await createShimServer(config);
  const shimPort = typeof shim.address() === "object" && shim.address() ? shim.address()!.port : 0;

  const response = await fetch(`http://127.0.0.1:${shimPort}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "aifchem-agent",
      messages: [{ role: "user", content: "我是谁？" }],
      stream: false,
    }),
  });

  assert.equal(response.status, 200);
  const json = (await response.json()) as any;
  assert.equal(json.choices[0].message.content, "You said your name is Xiaojin.");

  await new Promise<void>((resolve, reject) => shim.close((error) => (error ? reject(error) : resolve())));
  await new Promise<void>((resolve, reject) => upstream.server.close((error) => (error ? reject(error) : resolve())));
});

test("shim can preserve ui payloads and disclaimers when filters are disabled", async () => {
  const capture = { threadBodies: [] as any[], chatBodies: [] as any[] };
  const upstream = await createStatefulMockChatAppUpstream(capture, {
    onChat: (_count, body) => ({
      events: [
        {
          conversation_id: "conversation-123",
          thread_id: body.thread_id,
          workflow_id: body.workflow_id,
          message_id: "assistant-message-1",
          content:
            'You said your name is Xiaojin.\n\nThis answer is generated by a Large Language Model, please verify the information independently.\n\n```data {"data_type":"form","content":{"title":"Get In Touch"}}```',
        },
        { done: true },
      ],
    }),
  });

  const config: ShimConfig = {
    listen: { host: "127.0.0.1", port: 0 },
    upstream: {
      protocol: "chat_app_message_api",
      url: `http://127.0.0.1:${upstream.port}/api/message/chat`,
      method: "POST",
      chatApp: {
        conversationId: "conversation-123",
        workflowId: "workflow-456",
        createThreadPerRequest: true,
        filterUiPayloads: false,
        filterDisclaimers: false,
      },
    },
  };

  const shim = await createShimServer(config);
  const shimPort = typeof shim.address() === "object" && shim.address() ? shim.address()!.port : 0;

  const response = await fetch(`http://127.0.0.1:${shimPort}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "aifchem-agent",
      messages: [{ role: "user", content: "我是谁？" }],
      stream: false,
    }),
  });

  assert.equal(response.status, 200);
  const json = (await response.json()) as any;
  assert.match(json.choices[0].message.content, /This answer is generated by a Large Language Model/);
  assert.match(json.choices[0].message.content, /```data \{"data_type":"form"/);

  await new Promise<void>((resolve, reject) => shim.close((error) => (error ? reject(error) : resolve())));
  await new Promise<void>((resolve, reject) => upstream.server.close((error) => (error ? reject(error) : resolve())));
});
