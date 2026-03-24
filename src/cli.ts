import { ShimConfig } from "./aifchem/types.js";
import { parseArgs, readJson, resolveEnvPlaceholders, toAbsolute } from "./shared.js";
import { createShimServer } from "./shim/server.js";

async function main(): Promise<void> {
  const [command, ...rest] = process.argv.slice(2);
  const flags = parseArgs(rest);

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
  console.error("  tsx src/cli.ts serve-shim --config ./examples/shim.config.example.json");
  process.exitCode = 1;
}

await main();
