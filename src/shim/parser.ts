import { AifchemStreamEvent } from "../aifchem/types.js";

function safeJsonParse<T>(text: string): T | undefined {
  try {
    return JSON.parse(text) as T;
  } catch {
    return undefined;
  }
}

export function extractEventObjectsFromChunkBuffer(buffer: string): {
  rest: string;
  events: AifchemStreamEvent[];
} {
  const events: AifchemStreamEvent[] = [];
  let remaining = buffer;

  const sseBlocks = remaining.split("\n\n");
  if (sseBlocks.length > 1) {
    remaining = sseBlocks.pop() ?? "";
    for (const block of sseBlocks) {
      const dataLines = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim())
        .filter(Boolean);

      for (const data of dataLines) {
        if (data === "[DONE]") {
          events.push({ done: true });
          continue;
        }
        const parsed = safeJsonParse<AifchemStreamEvent>(data);
        if (parsed) {
          events.push(parsed);
        }
      }
    }
    return { rest: remaining, events };
  }

  const jsonLines = remaining.split("\n");
  if (jsonLines.length > 1) {
    remaining = jsonLines.pop() ?? "";
    for (const line of jsonLines.map((value) => value.trim()).filter(Boolean)) {
      const parsed = safeJsonParse<AifchemStreamEvent>(line);
      if (parsed) {
        events.push(parsed);
      }
    }
  }

  return { rest: remaining, events };
}

export function extractTextFromAifchemEvent(event: AifchemStreamEvent): {
  cumulative?: string;
  delta?: string;
  done?: boolean;
} {
  if (event.done) {
    return { done: true };
  }
  if (typeof event.delta === "string") {
    return { delta: event.delta };
  }
  if (typeof event.text === "string") {
    return { delta: event.text };
  }
  if (typeof event.content === "string") {
    return { cumulative: event.content };
  }
  return {};
}

