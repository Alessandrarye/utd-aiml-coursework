"""
Task Manager with User Authentication
A menu-driven application for managing personal tasks. Each user has their
own account (with a hashed password) and their own task list. Data persists
to a JSON file between sessions.

Author: Sandra Hershey-Zagrans
Course: UTD AI/ML Bootcamp - Unit 1 Assignment
"""

import json
import os
import hashlib

# Constants
DATA_FILE = "users.json"


# --- Persistence ---

def load_data(filename):
    """Read user/task data from JSON. Returns empty dict if file missing."""
    if not os.path.exists(filename):
        return {}
    
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded data for {len(data)} user(s).")
        return data
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading data: {e}")
        return {}


def save_data(data, filename):
    """Write user/task data to JSON."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except IOError as e:
        print(f"Error saving data: {e}")


# --- Authentication ---

def hash_password(password):
    """
    Return a SHA-256 hex digest of the password.
    
    Note: For production password storage, use bcrypt, scrypt, or Argon2.
    These provide salting and configurable work factors that defend against
    brute-force attacks. SHA-256 alone is used here for assignment scope.
    """
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def register(data):
    """Prompt for new username/password; add to data if username is free."""
    print("\n--- Register New Account ---")
    
    username = input("Choose a username: ").strip()
    
    if not username:
        print("Username cannot be empty.")
        return
    
    if username in data:
        print(f"Username '{username}' is already taken.")
        return
    
    password = input("Choose a password: ").strip()
    confirm = input("Confirm password: ").strip()
    
    if password != confirm:
        print("Passwords do not match. Registration cancelled.")
        return
    
    if not password:
        print("Password cannot be empty.")
        return
    
    data[username] = {
        "password_hash": hash_password(password),
        "tasks": [],
    }
    
    print(f"Account created for '{username}'. You can now log in.")


def login(data):
    """Prompt for credentials; return username on success, None on failure."""
    print("\n--- Login ---")
    
    username = input("Username: ").strip()
    password = input("Password: ").strip()
    
    if username not in data:
        print("Invalid username or password.")
        return None
    
    if data[username]["password_hash"] != hash_password(password):
        print("Invalid username or password.")
        return None
    
    print(f"Welcome back, {username}!")
    return username


# --- Task operations (all require an authenticated username) ---

def add_task(data, username):
    """Prompt for a task description and add it to the user's task list."""
    print("\n--- Add New Task ---")
    
    description = input("Task description: ").strip()
    
    if not description:
        print("Description cannot be empty. Task not added.")
        return
    
    user_tasks = data[username]["tasks"]
    
    # Generate next ID: max existing ID + 1, or 1 if no tasks yet
    if user_tasks:
        next_id = max(task["id"] for task in user_tasks) + 1
    else:
        next_id = 1
    
    new_task = {
        "id": next_id,
        "description": description,
        "status": "Pending",
    }
    
    user_tasks.append(new_task)
    print(f"Added task #{next_id}: {description}")


def view_tasks(data, username):
    """Display all tasks for the given user."""
    print(f"\n--- Tasks for {username} ---")
    
    user_tasks = data[username]["tasks"]
    
    if not user_tasks:
        print("No tasks yet. Add one from the menu.")
        return
    
    for task in user_tasks:
        status_marker = "✓" if task["status"] == "Completed" else " "
        print(f"  [{status_marker}] #{task['id']}: {task['description']} ({task['status']})")


def complete_task(data, username):
    """Prompt for a task ID and mark it Completed."""
    print("\n--- Mark Task as Completed ---")
    
    user_tasks = data[username]["tasks"]
    
    if not user_tasks:
        print("No tasks to update.")
        return
    
    task_id_input = input("Enter the task ID to mark completed: ").strip()
    
    try:
        task_id = int(task_id_input)
    except ValueError:
        print(f"'{task_id_input}' is not a valid ID.")
        return
    
    for task in user_tasks:
        if task["id"] == task_id:
            if task["status"] == "Completed":
                print(f"Task #{task_id} is already completed.")
            else:
                task["status"] = "Completed"
                print(f"Task #{task_id} marked as completed.")
            return
    
    print(f"No task found with ID {task_id}.")


def delete_task(data, username):
    """Prompt for a task ID and remove it from the user's list."""
    print("\n--- Delete Task ---")
    
    user_tasks = data[username]["tasks"]
    
    if not user_tasks:
        print("No tasks to delete.")
        return
    
    task_id_input = input("Enter the task ID to delete: ").strip()
    
    try:
        task_id = int(task_id_input)
    except ValueError:
        print(f"'{task_id_input}' is not a valid ID.")
        return
    
    for i, task in enumerate(user_tasks):
        if task["id"] == task_id:
            removed = user_tasks.pop(i)
            print(f"Deleted task #{task_id}: {removed['description']}")
            return
    
    print(f"No task found with ID {task_id}.")


# --- Menus ---

def auth_menu():
    """Print pre-login menu options."""
    print("\n--- Task Manager ---")
    print("1. Register")
    print("2. Login")
    print("3. Exit")


def task_menu(username):
    """Print post-login menu options."""
    print(f"\n--- Tasks for {username} ---")
    print("1. Add task")
    print("2. View tasks")
    print("3. Mark task as completed")
    print("4. Delete task")
    print("5. Logout")


def main():
    """Top-level loop: auth menu first, then task menu after login."""
    data = load_data(DATA_FILE)

    while True:
        auth_menu()
        choice = input("Enter your choice (1-3): ").strip()

        if choice == "1":
            register(data)
            save_data(data, DATA_FILE)
        elif choice == "2":
            username = login(data)
            if username:
                run_task_loop(data, username)
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def run_task_loop(data, username):
    """Inner loop that runs while a user is logged in."""
    while True:
        task_menu(username)
        choice = input("Enter your choice (1-5): ").strip()

        if choice == "1":
            add_task(data, username)
            save_data(data, DATA_FILE)
        elif choice == "2":
            view_tasks(data, username)
        elif choice == "3":
            complete_task(data, username)
            save_data(data, DATA_FILE)
        elif choice == "4":
            delete_task(data, username)
            save_data(data, DATA_FILE)
        elif choice == "5":
            print(f"Logged out: {username}")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 5.")


if __name__ == "__main__":
    main()