# Redteaming AI FChem Agent

这个仓库现在只保留两条主线：

1. 把 AIFChem 的真实网页聊天接口适配成 OpenAI `/v1/chat/completions`
2. 把后续 red-team / eval 流量统一打到这个 OpenAI 兼容入口

## 安装

```bash
npm install
```

## 启动本地 shim

```bash
set -a; source .env; set +a; npm run serve:shim
```

默认配置在 [examples/shim.config.example.json](examples/shim.config.example.json)。
环境变量模板在 [.env.example](.env.example)。

当前 shim 会：

- 先调用 `POST https://chat-app.aifchem.com/api/thread/create`
- 再调用 `POST https://chat-app.aifchem.com/api/message/chat`
- 把上游 SSE 转成 OpenAI 风格的非流式或流式 delta 输出

没有设置 `AIFCHEM_AUTHORIZATION` 时，进程会直接启动失败。

## 获取 `.env` 里的环境变量

推荐流程：

1. 登录 [chat.aifchem.com](https://chat.aifchem.com/)。
2. 打开浏览器开发者工具。
3. 进入 `Network` 面板。
4. 在聊天页实际发送一条消息。
5. 找到请求 `https://chat-app.aifchem.com/api/message/chat`。
6. 右键这个请求，选择 `Copy as cURL`。
7. 从 cURL 里提取：
   - `Authorization: Bearer ...` 对应 `AIFCHEM_AUTHORIZATION`
   - 请求体里的 `workflow_id` 对应 `AIFCHEM_WORKFLOW_ID`
8. 再找到请求 `https://chat-app.aifchem.com/api/thread/create`，或直接从 `api/message/chat` 的上下文里确认同一会话使用的 `conversation_id`，填到 `AIFCHEM_CONVERSATION_ID`。
9. 把它们写入本地 `.env`。

`.env` 示例：

```bash
AIFCHEM_AUTHORIZATION="Bearer <your-browser-jwt>"
AIFCHEM_CONVERSATION_ID="<your-conversation-id>"
AIFCHEM_WORKFLOW_ID="<your-workflow-id>"
```

注意：

- `.env` 已在 `.gitignore` 中，不要把真实值写回示例文件。

## 本地调用

非流式：

```bash
curl http://127.0.0.1:4010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-local-test' \
  -d '{
    "model": "aifchem-agent",
    "stream": false,
    "messages": [
      {"role": "user", "content": "hello"}
    ]
  }'
```

流式：

```bash
curl -N http://127.0.0.1:4010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-local-test' \
  -d '{
    "model": "aifchem-agent",
    "stream": true,
    "messages": [
      {"role": "user", "content": "hello"}
    ]
  }'
```

## 评测入口

通过 OpenAI 兼容入口运行评测：

```bash
npm run run:openai
```

配置示例在 [examples/openai-target.example.json](examples/openai-target.example.json)，场景示例在 [examples/scenarios.safe.json](examples/scenarios.safe.json)。

LiteLLM 示例配置在 [examples/litellm.config.yaml](examples/litellm.config.yaml)。

## 额外文件

- 协议整理见 [docs/aifchem-agent-protocol.md](docs/aifchem-agent-protocol.md)
- 简单 Python 客户端见 [examples/python_chat_client.py](examples/python_chat_client.py)
