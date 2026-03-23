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

`content` 在真实站点上表现为 token 级 delta，而不是累计全文。复用同一个固定 `thread_id` 时，上游会返回 `There are unfinished messages in the current thread`，所以 shim 默认按请求自动创建新 thread。

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

1. 用配置中的 `conversationId` 调 `/api/thread/create`
2. 取返回的 `data.id` 作为 `thread_id`
3. 用最后一条 `user` 消息内容填充 `query`
4. 把上游 SSE 转成 OpenAI `chat.completion.chunk`

OpenAI 流式输出示例：

```text
data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...,"choices":[{"delta":{"role":"assistant"},"index":0}]}
data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...,"choices":[{"delta":{"content":"Hello"},"index":0}]}
data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...,"choices":[{"delta":{},"finish_reason":"stop","index":0}]}
data: [DONE]
```
