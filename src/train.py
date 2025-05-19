import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import joblib

os.environ['LOGNAME'] = "rania"

def main():
    # Build the path to the CSV file inside the dataset folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "..", "dataset", "Churn_Modelling.csv")

    # Check if the file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File '{csv_path}' not found. Please check the path.")

    # Load the dataset
    df = pd.read_csv(csv_path)

    # Drop unnecessary columns
    X = df.drop(columns=["Exited", "RowNumber", "CustomerId", "Surname"])
    y = df["Exited"]

    # One-hot encoding for categorical variables
    X = pd.get_dummies(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Set the MLflow experiment name
    mlflow.set_experiment("Bank_Churn_Prediction")

    # Start an MLflow run
    with mlflow.start_run():
        # Initialize and train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Test Accuracy: {acc:.4f}")

        mlflow.log_metric("accuracy", acc)

        # Fix schema warning: convert int columns to float64 for input_example
        sample_input = X_train.head(5).copy()
        int_cols = sample_input.select_dtypes(include=['int']).columns
        sample_input[int_cols] = sample_input[int_cols].astype('float64')

        mlflow.sklearn.log_model(model, "model", input_example=sample_input)

        joblib.dump(model, "model.pkl")
        joblib.dump(X_train.columns.tolist(), "columns.pkl")

if __name__ == "__main__":
    main()
