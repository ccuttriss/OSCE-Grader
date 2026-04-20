# TODO

## Auth / Identity

- **SAML / SSO adapter.** The current auth is email + scrypt-hashed
  password stored in the SQLite `users` table. Long-term we want SAML
  (or OIDC) so passwords never live in our DB. Replacement points:
  - `scripts/identity.py::sign_in` — swap for an SSO redirect.
  - `scripts/identity.py::complete_password_change` — delete once
    SSO is the only path (passwords go away).
  - `scripts/database.py::users` table — keep `email` + `role` columns,
    drop `password_hash` / `must_change_password`.
  - `app.py` sign-in form — replace with "Continue with SSO" button.

## Passwords (interim, pre-SSO)

- Self-service password change for non-bootstrap users is intentionally
  out of scope for now. All password resets go through an admin in the
  **Users** tab.
- No account lockout on repeated wrong passwords yet. Reconsider before
  opening the app beyond a trusted audience.
