import type { Request, Response, NextFunction } from 'express';
import { getUserById } from '../db/queries.js';

export async function requireAuth(req: Request, _res: Response, next: NextFunction) {
  const userId = req.headers['x-user-id'] as string;
  if (!userId) return next(new Error('Missing x-user-id header'));
  const user = await getUserById(userId);
  if (!user) return next(new Error('User not found'));
  (req as any).user = user;
  next();
}
