import { pool } from './pool.js';
import type { User, GetUserByIdResult } from '../models/user.js';

/**
 * getUserById — fetch a single user by their unique ID.
 * Returns null if no user found.
 */
export async function getUserById(id: string): Promise<GetUserByIdResult> {
  const result = await pool.query('SELECT * FROM users WHERE id = $1', [id]);
  return result.rows[0] ?? null;
}

export async function listUsers(): Promise<User[]> {
  const result = await pool.query('SELECT * FROM users ORDER BY created_at');
  return result.rows;
}
