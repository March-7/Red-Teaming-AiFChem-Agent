# AIFChem Agent Protocol

日期：`2026-03-23`

本文档只保留当前真正落地到 shim 的真实网页聊天协议。

## 真实上游接口

基于 HAR 观察，网页聊天链路是两步：

1. `POST https://chat-app.aifchem.com/api/thread/create`
2. `POST https://chat-app.aifchem.com/api/message/chat`

`thread_id` 不是任意 UUID，而是第 1 步创建后返回的资源 id。

## Step 1: Create Thread

请求：

```http
POST /api/thread/create
Content-Type: application/json
Authorization: Bearer <browser-jwt>
```

```json
{
  "conversation_id": "<conversation-id>"
}
```

响应示例：

```json
{
  "data": {
    "id": "<thread-id>",
    "conversation_id": "<conversation-id>"
  },
  "code": 200,
  "msg": ""
}
```

## Step 2: Chat

请求：

```http
POST /api/message/chat
Content-Type: application/json
Authorization: Bearer <browser-jwt>
Accept: */*
```

```json
{
  "thread_id": "<thread-id>",
  "workflow_id": "<workflow-id>",
  "parent_message_id": "",
  "query": "How to use AiFChem Agent?",
  "inputs": {}
}
```

响应类型：

```http
Content-Type: text/event-stream
```

事件示例：

```text
data: {"conversation_id":"...","thread_id":"...","workflow_id":"...","message_id":"...","content":"I","data_type":"markdown","metadata":{"status":"running"}}
```

`content` 在真实站点上表现为 token 级 delta，而不是累计全文。历史上直接复用同一个固定 `thread_id` 时，上游可能返回 `There are unfinished messages in the current thread`，所以 shim 不能只依赖“固定 thread + 最后一条 user 文本”。

## Shim 映射

OpenAI 输入：

```json
{
  "model": "aifchem-agent",
  "messages": [
    {
      "role": "user",
      "content": "hello"
    }
  ],
  "stream": true
}
```

shim 行为：

1. 如果请求里带了 `metadata.session_id`，shim 会优先读取这个 session 上次成功响应保存的 `thread_id / parent_message_id / workflow_id / conversation_id`
2. 如果当前没有可复用 thread，shim 会用 `conversationId` 调 `/api/thread/create`
3. 把完整 OpenAI `messages` 序列化成一个单字符串 `query`，而不是只取最后一条 `user`
4. 把上游 SSE 转成 OpenAI `chat.completion.chunk`
5. 从上游 SSE 提取最新的 `thread_id / message_id / workflow_id / conversation_id`，写回 shim 内部 session 状态
6. 如果复用旧 thread 时收到 `There are unfinished messages in the current thread`，shim 会自动新建 thread，并用同一份 `messages` 重新发起请求

## 推荐调用方式

为了让“多轮对话”和“多步工具调用 agent”都尽量稳，推荐调用方每轮都同时提供：

1. 完整的 OpenAI `messages` 历史
2. 稳定的 `metadata.session_id`

这样 shim 既可以优先复用真实上游 thread，也可以在 thread 续用失败时退回到“新 thread + 完整历史重放”。

OpenAI 流式输出示例：

```text
data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...,"choices":[{"delta":{"role":"assistant"},"index":0}]}
data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...,"choices":[{"delta":{"content":"Hello"},"index":0}]}
data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...,"choices":[{"delta":{},"finish_reason":"stop","index":0}]}
data: [DONE]
```
