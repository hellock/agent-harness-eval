export interface User {
  id: string;
  name: string;
  email: string;
  role: 'admin' | 'member';
  createdAt: Date;
}

export type GetUserByIdResult = User | null;

export interface UserRepository {
  getUserById(id: string): Promise<GetUserByIdResult>;
  listUsers(): Promise<User[]>;
  deleteUser(id: string): Promise<boolean>;
}
