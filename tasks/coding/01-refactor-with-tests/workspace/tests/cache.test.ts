import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { CacheManager } from '../src/cache.js';

describe('CacheManager', () => {
  let cache: CacheManager;
  beforeEach(() => { cache = new CacheManager(3); vi.useFakeTimers(); });
  afterEach(() => { vi.useRealTimers(); });

  it('stores and retrieves values', () => {
    cache.set('a', 42, 10000);
    expect(cache.get('a')).toBe(42);
  });

  it('returns undefined for missing keys', () => {
    expect(cache.get('missing')).toBeUndefined();
  });

  it('expires entries after TTL', () => {
    cache.set('a', 'val', 1000);
    vi.advanceTimersByTime(1001);
    expect(cache.get('a')).toBeUndefined();
  });

  it('deletes entries', () => {
    cache.set('a', 1, 5000);
    expect(cache.delete('a')).toBe(true);
    expect(cache.get('a')).toBeUndefined();
    expect(cache.delete('a')).toBe(false);
  });

  it('reports correct size', () => {
    cache.set('a', 1, 5000);
    cache.set('b', 2, 5000);
    expect(cache.size).toBe(2);
  });

  it('evicts oldest when full', () => {
    cache.set('a', 1, 10000);
    cache.set('b', 2, 10000);
    cache.set('c', 3, 10000);
    cache.set('d', 4, 10000); // should evict 'a'
    expect(cache.get('a')).toBeUndefined();
    expect(cache.get('d')).toBe(4);
    expect(cache.size).toBe(3);
  });

  // True LRU: recently accessed entries should NOT be evicted next.
  // A FIFO implementation will fail this test.
  it('evicts least RECENTLY USED, not least recently inserted', () => {
    cache.set('a', 1, 10000);
    cache.set('b', 2, 10000);
    cache.set('c', 3, 10000);
    // Touch 'a' so it becomes most-recently-used.
    expect(cache.get('a')).toBe(1);
    // Inserting 'd' should now evict 'b' (oldest unused), not 'a'.
    cache.set('d', 4, 10000);
    expect(cache.get('a')).toBe(1);
    expect(cache.get('b')).toBeUndefined();
    expect(cache.get('c')).toBe(3);
    expect(cache.get('d')).toBe(4);
  });

  it('clears all entries', () => {
    cache.set('a', 1, 5000);
    cache.set('b', 2, 5000);
    cache.clear();
    expect(cache.size).toBe(0);
  });
});
