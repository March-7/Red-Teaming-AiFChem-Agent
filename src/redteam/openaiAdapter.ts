import { AdapterResponse, Scenario, TargetAdapter } from "./types.js";
import { OpenAICompatibleConfig } from "../aifchem/types.js";

export class OpenAICompatibleAdapter implements TargetAdapter {
  public readonly name = "openai-compatible";
  private readonly config: OpenAICompatibleConfig;

  public constructor(config: OpenAICompatibleConfig) {
    this.config = config;
  }

  public async run(prompt: string, _scenario: Scenario): Promise<AdapterResponse> {
    const url = new URL(this.config.path ?? "/v1/chat/completions", this.config.baseUrl).toString();
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(this.config.apiKey ? { Authorization: `Bearer ${this.config.apiKey}` } : {}),
      },
      body: JSON.stringify({
        model: this.config.model,
        stream: this.config.stream ?? false,
        messages: [
          {
            role: "user",
            content: prompt,
          },
        ],
      }),
    });

    if (!response.ok) {
      throw new Error(`OpenAI-compatible request failed: ${response.status} ${response.statusText}`);
    }

    if (this.config.stream === true) {
      const text = await response.text();
      const content = [...text.matchAll(/"content":"([^"]*)"/g)]
        .map((match) => match[1] ?? "")
        .join("");

      return {
        rawText: content,
        meta: { adapter: "openai-compatible", stream: true },
      };
    }

    const json = (await response.json()) as {
      choices?: Array<{ message?: { content?: string } }>;
    };

    return {
      rawText: json.choices?.[0]?.message?.content ?? "",
      meta: { adapter: "openai-compatible", stream: false },
    };
  }
}

