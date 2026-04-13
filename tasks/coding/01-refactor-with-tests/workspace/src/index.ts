export { formatDate } from './utils.js';
export { parseConfig } from './parser.js';
export { log, warn, error, setLevel } from './logger.js';

async function main() {
  log('Application started');
  const config = parseConfig('{"port": 3000}');
  log(`Config loaded: port=${config.port}`);
}

main().catch(console.error);
