# Personal Expense Tracker

A menu-driven command-line application for logging, categorizing, and tracking personal expenses against a monthly budget. Expenses persist to a CSV file between sessions.

Built as Unit 1 Assignment for the UTD AI/ML Bootcamp.

## Features

- **Add expense** — log expenses with date, category, amount, and description
- **View expenses** — display all stored expenses in an aligned table; skips entries with missing fields
- **Track budget** — set a monthly budget and see total spent, remaining balance, or warning if exceeded
- **Persistence** — expenses save to `expenses.csv` and reload automatically on next run
- **Input validation** — graceful handling of invalid amounts; auto-save on exit

## Requirements

- Python 3.x (uses only the standard library: `csv`, `os`)

## How to Run

From this directory:

```bash
python expense_tracker.py
```

The program displays a menu and prompts for your choice (1–5):

```
--- Personal Expense Tracker ---
1. Add expense
2. View expenses
3. Track budget
4. Save expenses
5. Exit
```

## Sample Interaction

```
--- Personal Expense Tracker ---
1. Add expense
2. View expenses
3. Track budget
4. Save expenses
5. Exit
Enter your choice (1-5): 1

--- Add New Expense ---
Date (YYYY-MM-DD): 2026-05-21
Category (e.g., Food, Travel): Food
Amount: 15.50
Description: Lunch
Added: Food - $15.50 on 2026-05-21
```

## Data Format

Expenses are stored in `expenses.csv` with the following columns:

```
date,category,amount,description
2026-05-21,Food,15.50,Lunch
```

The CSV file is created automatically on first save and ignored by git (see `.gitignore`).

## Design Notes

- **Data structure**: list of dictionaries, where each expense is `{"date", "category", "amount", "description"}`. Chosen for self-documenting access and direct compatibility with `csv.DictReader`/`DictWriter`.
- **Type conversion**: amounts are explicitly cast to `float` on load, since CSV stores everything as strings.
- **Error handling**: invalid amount input does not crash the program; the user is prompted again at the menu.

## Author

Sandra Hershey-Zagrans — UTD AI/ML Bootcamp, Unit 1