export type ExpectedBehavior =
  | "refuse_or_redirect"
  | "protect_system_prompt"
  | "protect_sensitive_data"
  | "answer_normally";

export interface Scenario {
  id: string;
  category: string;
  prompt: string;
  expectedBehavior: ExpectedBehavior;
}

export interface AdapterResponse {
  rawText: string;
  meta?: Record<string, string | number | boolean>;
}

export interface TargetAdapter {
  readonly name: string;
  run(prompt: string, scenario: Scenario): Promise<AdapterResponse>;
  close?(): Promise<void>;
}

export interface ScenarioResult {
  scenarioId: string;
  category: string;
  expectedBehavior: ExpectedBehavior;
  verdict: "pass" | "fail";
  signals: string[];
  responsePreview: string;
  elapsedMs: number;
}
