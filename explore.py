# explore.py

import pandas as pd  # pandas = Excel for Python. Loads and manipulates tabular data.

# Load the CSV into a DataFrame (think of it like a table/spreadsheet in memory)
df = pd.read_csv('train.csv')

# ── 1. FIRST LOOK ──────────────────────────────────────────────────────────────

print("=== Shape (rows, columns) ===")
print(df.shape)
# Output: (891, 12) → 891 passengers, 12 columns

print("\n=== First 5 rows ===")
print(df.head())
# Shows you what the actual data looks like

print("\n=== Column names ===")
print(df.columns.tolist())
# ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 
#  'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

# ── 2. UNDERSTAND EACH COLUMN ─────────────────────────────────────────────────
# PassengerId  → just an ID number, useless for prediction
# Survived     → 0 or 1 — THIS is what we're trying to predict (the "label")
# Pclass       → ticket class: 1=first, 2=second, 3=third
# Name         → passenger name (useless for ML, too unique)
# Sex          → male/female
# Age          → age in years
# SibSp        → number of siblings/spouses aboard
# Parch        → number of parents/children aboard
# Ticket       → ticket number (useless)
# Fare         → how much they paid
# Cabin        → cabin number (mostly missing)
# Embarked     → port of embarkation: C=Cherbourg, Q=Queenstown, S=Southampton

# ── 3. CHECK FOR MISSING DATA ─────────────────────────────────────────────────
print("\n=== Missing values per column ===")
print(df.isnull().sum())
# Age: 177 missing ← we need to handle this
# Cabin: 687 missing ← way too many missing, we'll drop this column
# Embarked: 2 missing ← easy to fix

# ── 4. BASIC STATISTICS ───────────────────────────────────────────────────────
print("\n=== Basic stats ===")
print(df.describe())
# Shows min, max, mean, std for numeric columns
# Key insight: mean of Survived = 0.38 → only 38% survived

# ── 5. SURVIVAL PATTERNS ──────────────────────────────────────────────────────
print("\n=== Survival rate by Sex ===")
print(df.groupby('Sex')['Survived'].mean())
# female    0.74  → 74% of women survived
# male      0.19  → only 19% of men survived
# Sex is a VERY strong predictor!

print("\n=== Survival rate by Pclass ===")
print(df.groupby('Pclass')['Survived'].mean())
# 1    0.63  → 63% of 1st class survived
# 2    0.47
# 3    0.24  → only 24% of 3rd class survived
# Class matters a lot too