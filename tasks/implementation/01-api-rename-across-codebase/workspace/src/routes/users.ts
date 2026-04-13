import { Router } from 'express';
import { getUserById } from '../db/queries.js';
import { fetchUserProfile, isAdmin } from '../services/user-service.js';

const router = Router();

// GET /users/:id — public profile
router.get('/:id', async (req, res) => {
  const profile = await fetchUserProfile(req.params.id);
  if (!profile) return res.status(404).json({ error: 'User not found' });
  res.json(profile);
});

// GET /users/:id/admin-check — internal
router.get('/:id/admin-check', async (req, res) => {
  // Direct DB call for performance (skip service layer)
  const user = await getUserById(req.params.id);
  res.json({ isAdmin: user?.role === 'admin' });
});

export default router;
