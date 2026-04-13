# User API

## Database Layer

### `getUserById(id: string): Promise<GetUserByIdResult>`
Fetches a user by ID. Returns `null` if not found.

### `listUsers(): Promise<User[]>`
Returns all users sorted by creation date.

## Service Layer

### `fetchUserProfile(userId: string)`
Calls `getUserById` internally, strips sensitive fields.

### `isAdmin(userId: string)`
Checks if the user returned by `getUserById` has admin role.
