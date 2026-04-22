import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import pickle  # ← NEW: to also save a plain .pkl file outside MLflow

# ── LOAD & PREPROCESS (same as before) ────────────────────────────────────────
df = pd.read_csv('train.csv')
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── TRAIN ──────────────────────────────────────────────────────────────────────
n_estimators = 100
max_depth = 5

mlflow.set_experiment("titanic-survival")

with mlflow.start_run():
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "random-forest-model")

    print(f"Accuracy: {accuracy:.2%}")

    # ── NEW: also save plain pkl to /output (mounted volume) ──────────────────
    # /output is a folder we'll mount from the host when running the container
    # This lets the serving container pick up the model file
    os.makedirs("/output", exist_ok=True)
    with open("/output/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved to /output/model.pkl")