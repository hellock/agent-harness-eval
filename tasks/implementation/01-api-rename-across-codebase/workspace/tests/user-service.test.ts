import { describe, it, expect, vi } from 'vitest';

// Mock the getUserById function
vi.mock('../src/db/queries.js', () => ({
  getUserById: vi.fn(),
}));

import { fetchUserProfile, isAdmin } from '../src/services/user-service.js';
import { getUserById } from '../src/db/queries.js';

const mockGetUserById = getUserById as ReturnType<typeof vi.fn>;

describe('fetchUserProfile', () => {
  it('returns user profile when found', async () => {
    mockGetUserById.mockResolvedValue({ id: '1', name: 'Alice', email: 'a@b.c', role: 'admin', createdAt: new Date() });
    const result = await fetchUserProfile('1');
    expect(result).toBeDefined();
    expect(result!.name).toBe('Alice');
    expect(mockGetUserById).toHaveBeenCalledWith('1');
  });

  it('returns null when not found', async () => {
    mockGetUserById.mockResolvedValue(null);
    const result = await fetchUserProfile('999');
    expect(result).toBeNull();
  });
});

describe('isAdmin', () => {
  it('returns true for admin users', async () => {
    mockGetUserById.mockResolvedValue({ id: '1', role: 'admin' });
    expect(await isAdmin('1')).toBe(true);
  });

  it('returns false for non-admin', async () => {
    mockGetUserById.mockResolvedValue({ id: '2', role: 'member' });
    expect(await isAdmin('2')).toBe(false);
  });
});
