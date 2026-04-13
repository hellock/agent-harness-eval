// ERROR LAYER 1: same missing .js extension
import { Logger, LogLevel } from './types';

// ERROR LAYER 4 (revealed after fixing layers 1-3):
// Class implements Logger but log() signature doesn't match —
// interface requires meta?: Record<string, unknown>
// but implementation declares meta?: object, which is not assignable
// because Record<string, unknown> has an index signature that
// plain `object` does not provide.
class ConsoleLogger implements Logger {
  private prefix: string;

  constructor(prefix: string) {
    this.prefix = prefix;
  }

  log(level: LogLevel, message: string, meta?: object): void {
    const ts = new Date().toISOString();
    const metaStr = meta ? ' ' + JSON.stringify(meta) : '';
    console.log(`[${ts}] [${this.prefix}] ${level.toUpperCase()}: ${message}${metaStr}`);
  }
}

// ERROR LAYER 5 (revealed after fixing layer 4):
// createLogger returns Logger type but constructs ConsoleLogger
// then calls a method .setLevel() that doesn't exist on either type
export function createLogger(name: string, level: LogLevel = 'info'): Logger {
  const logger = new ConsoleLogger(name);
  logger.setLevel(level);
  return logger;
}
