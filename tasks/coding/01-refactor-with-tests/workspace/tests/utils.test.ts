import { describe, it, expect } from 'vitest';
import { formatDate, capitalize } from '../src/utils.js';

describe('formatDate', () => {
  it('formats a date correctly', () => {
    const date = new Date(2024, 0, 15);
    expect(formatDate(date)).toBe('2024-01-15');
  });
});

describe('capitalize', () => {
  it('capitalizes first letter', () => {
    expect(capitalize('hello')).toBe('Hello');
  });

  it('handles empty string', () => {
    expect(capitalize('')).toBe('');
  });
});
