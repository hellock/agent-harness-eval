export interface AppConfig {
  port: number;
  host?: string;
  debug?: boolean;
}

export function parseConfig(raw: string): AppConfig {
  const parsed = JSON.parse(raw);
  return {
    port: parsed.port ?? 3000,
    host: parsed.host ?? 'localhost',
    debug: parsed.debug ?? false,
  };
}
