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
      res.write('data: {"content":"Hello"}\n\n');
      res.write('data: {"content":" world"}\n\n');
      res.write('data: {"done":true}\n\n');
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
