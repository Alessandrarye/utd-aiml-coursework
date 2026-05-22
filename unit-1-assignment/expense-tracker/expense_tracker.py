"""
Personal Expense Tracker
A menu-driven application to log, categorize, and track personal expenses
against a monthly budget. Expenses persist to a CSV file between sessions.

Author: Sandra Hershey-Zagrans
Course: UTD AI/ML Bootcamp - Unit 1 Assignment
"""

import csv
import os

# Constants
EXPENSES_FILE = "expenses.csv"


def add_expense(expenses):
    """Prompt user for expense details and append to the expenses list."""
    print("\n--- Add New Expense ---")
    
    date = input("Date (YYYY-MM-DD): ").strip()
    category = input("Category (e.g., Food, Travel): ").strip()
    amount_input = input("Amount: ").strip()
    description = input("Description: ").strip()
    
    # Convert amount from string to float
    try:
        amount = float(amount_input)
    except ValueError:
        print(f"'{amount_input}' is not a valid number. Expense not added.")
        return
    
    # Build the expense dictionary
    expense = {
        "date": date,
        "category": category,
        "amount": amount,
        "description": description,
    }
    
    expenses.append(expense)
    print(f"Added: {category} - ${amount:.2f} on {date}")
    pass


def view_expenses(expenses):
    """Display all stored expenses, skipping incomplete entries."""
    print("\n--- All Expenses ---")
    
    if not expenses:
        print("No expenses recorded yet.")
        return
    
    required_fields = ("date", "category", "amount", "description")
    
    for i, expense in enumerate(expenses, start=1):
        # Validate that all required fields exist and are not empty
        if not all(expense.get(field) for field in required_fields):
            print(f"{i}. [Skipped — incomplete entry]")
            continue
        
        print(
            f"{i}. {expense['date']} | {expense['category']:<10} | "
            f"${expense['amount']:>8.2f} | {expense['description']}"
        )
    pass



    pass
def set_budget():
    """Prompt user for a monthly budget amount and return it."""
    print("\n--- Set Monthly Budget ---")
    
    while True:
        budget_input = input("Enter your monthly budget: $").strip()
        try:
            budget = float(budget_input)
            if budget < 0:
                print("Budget cannot be negative. Try again.")
                continue
            print(f"Monthly budget set to ${budget:.2f}")
            return budget
        except ValueError:
            print(f"'{budget_input}' is not a valid number. Try again.")


def track_budget(expenses, budget):
    """Compare total expenses against budget; warn if exceeded."""
    print("\n--- Budget Status ---")
    
    # Sum all valid expense amounts
    total = sum(
        expense["amount"]
        for expense in expenses
        if isinstance(expense.get("amount"), (int, float))
    )
    
    print(f"Monthly budget:    ${budget:.2f}")
    print(f"Total spent:       ${total:.2f}")
    
    if total > budget:
        overage = total - budget
        print(f"⚠️  You have exceeded your budget by ${overage:.2f}!")
    else:
        remaining = budget - total
        print(f"You have ${remaining:.2f} left for the month.")

def save_expenses(expenses, filename):
    """Write the expenses list to a CSV file."""
    pass


def load_expenses(filename):
    """Read expenses from a CSV file and return as a list of dicts."""
    # For now, just return an empty list. We'll implement CSV loading later.
    return []
    pass


def display_menu():
    """Print the menu options to the screen."""
    print("\n--- Personal Expense Tracker ---")
    print("1. Add expense")
    print("2. View expenses")
    print("3. Track budget")
    print("4. Save expenses")
    print("5. Exit")


def main():
    """Main program loop: load data, show menu, dispatch to handlers."""
    expenses = load_expenses(EXPENSES_FILE)
    budget = None

    while True:
        display_menu()
        choice = input("Enter your choice (1-5): ").strip()

        if choice == "1":
            add_expense(expenses)
        elif choice == "2":
            view_expenses(expenses)
        elif choice == "3":
            if budget is None:
                budget = set_budget()
            track_budget(expenses, budget)
        elif choice == "4":
            save_expenses(expenses, EXPENSES_FILE)
        elif choice == "5":
            save_expenses(expenses, EXPENSES_FILE)
            print("Expenses saved. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 5.")


if __name__ == "__main__":
    main()