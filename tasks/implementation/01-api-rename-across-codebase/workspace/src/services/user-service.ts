import { getUserById } from '../db/queries.js';
import type { User } from '../models/user.js';

export async function fetchUserProfile(userId: string): Promise<User | null> {
  const user = await getUserById(userId);
  if (!user) return null;
  // Strip sensitive fields before returning
  const { ...profile } = user;
  return profile;
}

export async function isAdmin(userId: string): Promise<boolean> {
  const user = await getUserById(userId);
  return user?.role === 'admin';
}
