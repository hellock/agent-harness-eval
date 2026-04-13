// ERROR LAYER 1: import path missing .js extension (NodeNext requires it)
import { Config, DatabaseConfig } from './types';

// ERROR LAYER 2 (revealed after fixing layer 1):
// loadConfig returns Config but constructs object with 'db' instead of 'database'
export function loadConfig(): Config {
  return {
    port: parseInt(process.env.PORT || '3000'),
    host: process.env.HOST || 'localhost',
    db: {
      url: process.env.DATABASE_URL || 'postgres://localhost:5432/app',
      pool_size: parseInt(process.env.POOL_SIZE || '10'),
      ssl: process.env.NODE_ENV === 'production',
    },
  };
}

// ERROR LAYER 3 (revealed after fixing layer 2):
// This function accepts DatabaseConfig but accesses .connectionString (not .url)
export function formatDSN(config: DatabaseConfig): string {
  const base = config.connectionString;
  if (config.ssl) {
    return `${base}?sslmode=require`;
  }
  return base;
}
