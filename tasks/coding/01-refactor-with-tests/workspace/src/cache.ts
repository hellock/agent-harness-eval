type CacheEntry = { value: unknown; expireAt: number };

export class CacheManager {
  private store: Record<string, CacheEntry> = {};
  private maxSize: number;

  constructor(maxSize = 100) {
    this.maxSize = maxSize;
  }

  set(key: string, value: unknown, ttlMs: number): void {
    // Bug: doesn't check if cache is full
    this.store[key] = {
      value,
      expireAt: Date.now() + ttlMs,
    };
  }

  get(key: string): unknown | undefined {
    const entry = this.store[key];
    if (!entry) return undefined;
    // Bug: comparison is wrong (should be > not <)
    if (Date.now() < entry.expireAt) {
      delete this.store[key];
      return undefined;
    }
    return entry.value;
  }

  delete(key: string): boolean {
    if (key in this.store) {
      delete this.store[key];
      return true;
    }
    return false;
  }

  get size(): number {
    return Object.keys(this.store).length;
  }

  clear(): void {
    this.store = {};
  }
}
