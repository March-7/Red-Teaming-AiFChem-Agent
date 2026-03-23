import { randomUUID } from "node:crypto";

export interface OpenAIStreamState {
  id: string;
  model: string;
  created: number;
}

export function createOpenAIStreamState(model: string): OpenAIStreamState {
  return {
    id: `chatcmpl-${randomUUID()}`,
    model,
    created: Math.floor(Date.now() / 1000),
  };
}

export function makeChunk(
  state: OpenAIStreamState,
  delta: Record<string, unknown>,
  finishReason: string | null,
): string {
  return JSON.stringify({
    id: state.id,
    object: "chat.completion.chunk",
    created: state.created,
    model: state.model,
    choices: [
      {
        index: 0,
        delta,
        finish_reason: finishReason,
      },
    ],
  });
}

export function makeCompletion(state: OpenAIStreamState, content: string): Record<string, unknown> {
  return {
    id: state.id,
    object: "chat.completion",
    created: state.created,
    model: state.model,
    choices: [
      {
        index: 0,
        finish_reason: "stop",
        message: {
          role: "assistant",
          content,
        },
      },
    ],
    usage: {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
    },
  };
}

