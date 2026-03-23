import { OpenAICompatibleAdapter } from "./redteam/openaiAdapter.js";
import { loadScenarios, runScenarios } from "./redteam/runner.js";
import { OpenAICompatibleConfig, ShimConfig } from "./aifchem/types.js";
import { parseArgs, readJson, resolveEnvPlaceholders, toAbsolute } from "./shared.js";
import { createShimServer } from "./shim/server.js";

async function main(): Promise<void> {
  const [command, ...rest] = process.argv.slice(2);
  const flags = parseArgs(rest);

  if (command === "run") {
    const scenariosPath = toAbsolute(String(flags.scenarios ?? "./examples/scenarios.safe.json"));
    const scenarios = await loadScenarios(scenariosPath);
    const configPath = flags.config
      ? toAbsolute(String(flags.config))
      : toAbsolute("./examples/openai-target.example.json");
    const config = await readJson<OpenAICompatibleConfig>(configPath);
    const adapter = new OpenAICompatibleAdapter(config);
    const report = await runScenarios({ adapter, scenarios });
    console.log(JSON.stringify(report, null, 2));
    return;
  }

  if (command === "serve-shim") {
    const configPath = flags.config
      ? toAbsolute(String(flags.config))
      : toAbsolute("./examples/shim.config.example.json");
    const config = resolveEnvPlaceholders(await readJson<ShimConfig>(configPath));
    await createShimServer(config);
    console.log(`shim listening on http://${config.listen.host}:${config.listen.port}`);
    return;
  }

  console.error("Usage:");
  console.error("  tsx src/cli.ts run --config ./examples/openai-target.example.json --scenarios ./examples/scenarios.safe.json");
  console.error("  tsx src/cli.ts serve-shim --config ./examples/shim.config.example.json");
  process.exitCode = 1;
}

await main();
