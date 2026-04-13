export interface Config {
  port: number;
  host: string;
  database: DatabaseConfig;
}

export interface DatabaseConfig {
  url: string;
  pool_size: number;
  ssl: boolean;
}

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface Logger {
  log(level: LogLevel, message: string, meta?: Record<string, unknown>): void;
}
