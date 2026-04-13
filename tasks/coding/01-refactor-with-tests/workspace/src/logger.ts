let currentLevel: 'debug' | 'info' | 'warn' | 'error' = 'info';

const levels = { debug: 0, info: 1, warn: 2, error: 3 };

export function setLevel(level: typeof currentLevel): void {
  currentLevel = level;
}

export function log(message: string): void {
  if (levels[currentLevel] <= levels.info) {
    console.log(`[INFO] ${message}`);
  }
}

export function warn(message: string): void {
  if (levels[currentLevel] <= levels.warn) {
    console.warn(`[WARN] ${message}`);
  }
}

export function error(message: string): void {
  console.error(`[ERROR] ${message}`);
}
