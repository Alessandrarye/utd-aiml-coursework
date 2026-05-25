# Task Manager with User Authentication

A menu-driven command-line application for managing personal tasks across multiple user accounts. Each user has their own login (with a hashed password) and their own private task list. Data persists to a JSON file between sessions.

Built as Unit 1 Assignment for the UTD AI/ML Bootcamp.

## Features

- **User registration** — create a new account with a unique username and confirmed password
- **Authentication** — log in with stored credentials; passwords are SHA-256 hashed, never stored as plaintext
- **Per-user task isolation** — each user only sees and manages their own tasks
- **Task operations** — add, view, mark as completed, and delete tasks by ID
- **Persistence** — all user data and tasks save automatically to `users.json` after every change
- **Two-state menu system** — separate auth menu (Register / Login / Exit) and task menu (Add / View / Complete / Delete / Logout)

## Requirements

- Python 3.x (uses only the standard library: `json`, `os`, `hashlib`)

## How to Run

From this directory:

```bash
python task_manager.py
```

The program opens with the authentication menu:

```
--- Task Manager ---
1. Register
2. Login
3. Exit
```

After successful login, the task menu becomes available:

```
--- Tasks for alice ---
1. Add task
2. View tasks
3. Mark task as completed
4. Delete task
5. Logout
```

## Sample Interaction

```
--- Task Manager ---
1. Register
2. Login
3. Exit
Enter your choice (1-3): 1

--- Register New Account ---
Choose a username: alice
Choose a password: hunter2
Confirm password: hunter2
Account created for 'alice'. You can now log in.

--- Task Manager ---
1. Register
2. Login
3. Exit
Enter your choice (1-3): 2

--- Login ---
Username: alice
Password: hunter2
Welcome back, alice!

--- Tasks for alice ---
1. Add task
2. View tasks
3. Mark task as completed
4. Delete task
5. Logout
Enter your choice (1-5): 1

--- Add New Task ---
Task description: Finish bootcamp assignment
Added task #1: Finish bootcamp assignment
```

## Data Format

User data and tasks are stored in `users.json` as a nested dictionary:

```json
{
  "alice": {
    "password_hash": "f52fbd32b2b3b86ff88ef6c490628285f482af15ddcb29541f94bcf526a3f6c7",
    "tasks": [
      {
        "id": 1,
        "description": "Finish bootcamp assignment",
        "status": "Pending"
      }
    ]
  }
}
```

The JSON file is created automatically on first save and ignored by git (see `.gitignore`).

## Design Notes

### Data structure: nested dictionary

Users are keyed by username at the top level, and each user contains a `password_hash` field and a list of `tasks`. This was chosen over alternatives (flat lists with username foreign keys, separate files per user) because:

- **Direct O(1) lookup** of a user's tasks: `data[username]["tasks"]`
- **Natural mapping to JSON** without requiring joins or filtering
- **Per-user isolation is enforced by the access pattern** — there is no operation that crosses user boundaries

### Password hashing

Passwords are hashed with SHA-256 from Python's built-in `hashlib`. The hex digest (a 64-character string) is stored in `users.json`; the original password is never written to disk. Login works by hashing the user's attempt and comparing to the stored hash — the original password is never retrieved or decrypted (it cannot be; hashing is one-way).

**Limitations of this approach (acknowledged for transparency):**

- **No salting** — two users with the same password produce identical hashes, making rainbow table attacks easier
- **SHA-256 is fast** — fast hashing benefits brute-force attackers, since they can try billions of candidates per second on modern hardware
- **Plaintext input** — `input()` echoes the password as the user types it

For production password storage, the appropriate algorithms are `bcrypt`, `scrypt`, or `Argon2` (all of which include salting and configurable work factors), combined with `getpass.getpass()` for masked input. SHA-256 alone is used here to satisfy the assignment scope without external dependencies.

### Uniform error messages on login

Both "username does not exist" and "wrong password" return the same message: `Invalid username or password.` This prevents an attacker from enumerating valid usernames by probing the system — a common pattern called username enumeration. Every login form on a security-conscious site behaves this way.

### Stable task IDs

When a task is deleted, its ID is not reused. New tasks always get `max(existing_ids) + 1`. This matches how relational databases handle primary keys and preserves the user's mental model of "task 3 is the grocery run" even after deletes.

### Auto-save on mutation

Unlike the companion Expense Tracker (which has an explicit Save menu option), this program saves after every mutation (register, add task, complete task, delete task). The cost is small (one JSON file write); the benefit is that no data is lost if the program is killed unexpectedly.

## State Machine

The program has two states with distinct menus:

```
                  ┌──────────────────┐
                  │   Auth Menu      │
                  │  1. Register     │
                  │  2. Login        │◄─── (Logout)
                  │  3. Exit         │
                  └────────┬─────────┘
                           │ login success
                           ▼
                  ┌──────────────────┐
                  │   Task Menu      │
                  │  1. Add task     │
                  │  2. View tasks   │
                  │  3. Complete     │
                  │  4. Delete       │
                  │  5. Logout       │
                  └──────────────────┘
```

The username obtained at login is threaded through to every task operation as a parameter, ensuring all operations are scoped to the authenticated user.

## Author

Sandra Hershey-Zagrans — UTD AI/ML Bootcamp, Unit 1