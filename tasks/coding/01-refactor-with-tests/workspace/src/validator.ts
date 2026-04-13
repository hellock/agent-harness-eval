export function validateEmail(email: string): boolean {
  return email.includes('@');
}

export function validatePort(port: number): boolean {
  return port >= 1 && port <= 65535;
}
