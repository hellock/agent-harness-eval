// Re-exports for external consumers
export { getUserById, listUsers } from './db/queries.js';
export type { User, GetUserByIdResult, UserRepository } from './models/user.js';
export { fetchUserProfile, isAdmin } from './services/user-service.js';
