import { randomUUID } from "node:crypto";
import { AifchemMessage, AifchemRequest, OpenAIChatCompletionRequest, OpenAIChatMessage, ShimConfig } from "./types.js";

function normalizeContent(content: OpenAIChatMessage["content"]): string {
  if (typeof content === "string") {
    return content;
  }

  return content
    .map((part) => {
      if (part.type === "text") {
        return part.text ?? "";
      }
      return `[${part.type}]`;
    })
    .join("\n")
    .trim();
}

function toPreservedMessages(messages: OpenAIChatMessage[]): AifchemMessage[] {
  return messages.map((message) => ({
    role: message.role,
    content: normalizeContent(message.content),
  }));
}

function toUserOnlyMessages(messages: OpenAIChatMessage[]): AifchemMessage[] {
  return messages.map((message) => ({
    role: "user",
    content: `${message.role.toUpperCase()}: ${normalizeContent(message.content)}`,
  }));
}

export function mapOpenAIRequestToAifchem(
  request: OpenAIChatCompletionRequest,
  config: ShimConfig,
): AifchemRequest {
  const roleMode = config.upstream.messageRoleMode ?? "preserve";
  const mappedMessages =
    roleMode === "user_only_concat"
      ? toUserOnlyMessages(request.messages)
      : toPreservedMessages(request.messages);

  const metadataContextLink =
    typeof request.metadata?.context_link === "string" ? request.metadata.context_link : undefined;

  return {
    message_id: randomUUID(),
    messages: mappedMessages,
    context_link: metadataContextLink ?? config.upstream.contextLink ?? "",
    response_mode: "streaming",
    ...(config.upstream.extraBody ?? {}),
  };
}

function getLastUserText(messages: OpenAIChatMessage[]): string {
  const lastUser = [...messages].reverse().find((message) => message.role === "user");
  return lastUser ? normalizeContent(lastUser.content) : "";
}

function formatChatAppTranscript(messages: OpenAIChatMessage[]): string {
  return messages
    .map((message) => {
      const content = normalizeContent(message.content);
      return `${message.role.toUpperCase()}:\n${content || "(empty)"}`;
    })
    .join("\n\n");
}

export function buildChatAppQuery(messages: OpenAIChatMessage[]): string {
  if (messages.length <= 1) {
    return getLastUserText(messages);
  }

  return [
    "The following is the full conversation transcript so far.",
    formatChatAppTranscript(messages),
    "Continue the conversation as the assistant and answer the latest user message using the prior assistant and tool context when relevant.",
  ].join("\n\n");
}

function getMetadataString(request: OpenAIChatCompletionRequest, key: string): string | undefined {
  const value = request.metadata?.[key];
  return typeof value === "string" && value ? value : undefined;
}

export function mapOpenAIRequestToChatAppMessageApi(
  request: OpenAIChatCompletionRequest,
  config: ShimConfig,
  overrides: {
    threadId?: string;
    workflowId?: string;
    parentMessageId?: string;
  } = {},
): Record<string, unknown> {
  const chatApp = config.upstream.chatApp;
  if (!chatApp) {
    throw new Error("Missing upstream.chatApp configuration");
  }

  const threadId = overrides.threadId ?? getMetadataString(request, "thread_id") ?? chatApp.threadId;
  if (!threadId) {
    throw new Error("Missing chat_app_message_api thread id");
  }

  return {
    thread_id: threadId,
    workflow_id: overrides.workflowId ?? getMetadataString(request, "workflow_id") ?? chatApp.workflowId,
    parent_message_id:
      overrides.parentMessageId ?? getMetadataString(request, "parent_message_id") ?? chatApp.parentMessageId ?? "",
    query: buildChatAppQuery(request.messages),
    inputs: chatApp.inputs ?? {},
  };
}
