import path from "node:path";
import { mkdir, readFile, writeFile } from "node:fs/promises";

export function utcTimestamp(): string {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

export async function ensureDir(dirPath: string): Promise<void> {
  await mkdir(dirPath, { recursive: true });
}

export async function writeText(filePath: string, content: string): Promise<void> {
  await ensureDir(path.dirname(filePath));
  await writeFile(filePath, content, "utf8");
}

export async function writeJson(filePath: string, data: unknown): Promise<void> {
  await writeText(filePath, `${JSON.stringify(data, null, 2)}\n`);
}

export async function readJson<T>(filePath: string): Promise<T> {
  const raw = await readFile(filePath, "utf8");
  return JSON.parse(raw) as T;
}

export function parseArgs(argv: string[]): Record<string, string | boolean> {
  const parsed: Record<string, string | boolean> = {};

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) {
      continue;
    }

    const key = token.slice(2);
    const next = argv[index + 1];
    if (!next || next.startsWith("--")) {
      parsed[key] = true;
      continue;
    }

    parsed[key] = next;
    index += 1;
  }

  return parsed;
}

export function toAbsolute(filePath: string): string {
  return path.isAbsolute(filePath) ? filePath : path.resolve(process.cwd(), filePath);
}

export function snippet(text: string, max = 280): string {
  return text.length <= max ? text : `${text.slice(0, max)}...`;
}

export function resolveEnvPlaceholders<T>(value: T): T {
  if (typeof value === "string" && value.startsWith("os.environ/")) {
    const envName = value.slice("os.environ/".length);
    const resolved = process.env[envName];
    if (resolved === undefined) {
      throw new Error(`Missing required environment variable: ${envName}`);
    }
    return resolved as T;
  }

  if (Array.isArray(value)) {
    return value.map((item) => resolveEnvPlaceholders(item)) as T;
  }

  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, nested]) => [key, resolveEnvPlaceholders(nested)]),
    ) as T;
  }

  return value;
}
