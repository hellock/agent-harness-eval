// ERROR LAYER 1: same missing .js extension (all 3 imports)
import { loadConfig, formatDSN } from './config';
import { createLogger } from './logger';
import { Config } from './types';

export function startApp(): void {
  const config = loadConfig();
  const logger = createLogger('app');

  logger.log('info', 'Starting application', { port: config.port });

  // This uses formatDSN correctly (after layer 3 fix)
  const dsn = formatDSN(config.database);
  logger.log('info', `Database: ${dsn}`);

  logger.log('info', `Listening on ${config.host}:${config.port}`);
}
