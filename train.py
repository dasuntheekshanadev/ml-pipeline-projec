# train.py

import pandas as pd
import mlflow                          # experiment tracker
import mlflow.sklearn                  # MLflow plugin for scikit-learn models
from sklearn.ensemble import RandomForestClassifier  # the ML algorithm
from sklearn.model_selection import train_test_split # splits data into train/test
from sklearn.metrics import accuracy_score           # measures how good the model is
from sklearn.preprocessing import LabelEncoder       # converts text → numbers

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
df = pd.read_csv('train.csv')

# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────
# "Features" = the input columns we feed to the model
# "Label"    = the column we want to predict (Survived)

# Drop columns that are useless for prediction
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Fix missing values
# Age: fill missing with the median age (middle value)
df['Age'] = df['Age'].fillna(df['Age'].median())

# Embarked: fill 2 missing rows with the most common port
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Convert text columns to numbers — ML models only understand numbers
# Sex: male→1, female→0
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])        # male=1, female=0
df['Embarked'] = le.fit_transform(df['Embarked'])  # C=0, Q=1, S=2

# ── 3. SPLIT INTO FEATURES (X) AND LABEL (y) ─────────────────────────────────
# X = everything the model uses as INPUT
# y = what the model is trying to PREDICT
X = df.drop('Survived', axis=1)   # all columns except Survived
y = df['Survived']                 # just the Survived column

print("Features (X) shape:", X.shape)   # (891, 7)
print("Label (y) shape:", y.shape)      # (891,)

# ── 4. SPLIT INTO TRAINING AND TEST SETS ─────────────────────────────────────
# We train on 80% of the data, then test on the remaining 20%
# The model NEVER sees the test set during training — it's like an exam
# This tells us if the model generalizes or just memorized the training data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% goes to test
    random_state=42     # fixed seed = reproducible split every run
)

print("Training rows:", len(X_train))   # ~712
print("Test rows:", len(X_test))        # ~179

# ── 5. CHOOSE AN ALGORITHM: RANDOM FOREST ────────────────────────────────────
# Random Forest = builds many decision trees and votes on the answer
# Decision tree logic: "Is sex female? → yes → did they pay > $30? → yes → survived"
# Random Forest builds 100 of these trees and takes a majority vote
# It's robust, works well without much tuning, great for beginners

n_estimators = 100   # number of trees
max_depth = 5        # how deep each tree can go (prevents overfitting)

# ── 6. TRAIN WITH MLFLOW TRACKING ─────────────────────────────────────────────
# mlflow.start_run() opens a "recording session"
# Everything inside gets logged to MLflow's local database
mlflow.set_experiment("titanic-survival")  # group runs under this experiment name

with mlflow.start_run():

    # Log the settings (hyperparameters) we're using this run
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("test_size", 0.2)

    # TRAIN THE MODEL
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)   # ← this is where learning happens

    # EVALUATE: run the model on the test set it's never seen
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel accuracy: {accuracy:.2%}")
    # Typically ~82% → model correctly predicts 82% of passengers

    # Log the result metric
    mlflow.log_metric("accuracy", accuracy)

    # SAVE THE MODEL — this is your "model artifact"
    # MLflow saves the model file + metadata + environment info
    mlflow.sklearn.log_model(model, "random-forest-model")

    print("\nRun logged to MLflow!")
    print("Run: mlflow ui   to see the dashboard")