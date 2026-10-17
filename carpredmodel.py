import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load dataset
df = pd.read_csv("car_price_prediction_.csv")


# 2. Drop Car ID if present
if "Car ID" in df.columns:
    df = df.drop(columns=["Car ID"])


# 3. Separate features and target
X = df.drop(columns=["Price"])
y = df["Price"]


# 4. Identify categorical and numerical columns
categorical_cols = [
    "Brand",
    "Fuel Type",
    "Transmission",
    "Condition",
    "Model"
]

numerical_cols = [
    col for col in X.columns
    if col not in categorical_cols
]
numeric_cols = [
    "Year",
    "Engine Size",
    "Mileage",
    "Price"
]

corr = df[numeric_cols].corr()

# 5. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols
        ),
        (
            "num",
            StandardScaler(),
            numerical_cols
        )
    ]
)


# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 7. Baseline model
baseline = DummyRegressor(strategy="mean")

baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)

print("\n===== BASELINE =====")
print("MAE:", mean_absolute_error(y_test, baseline_pred))
print("MSE:", mean_squared_error(y_test, baseline_pred))
print("R2 Score:", r2_score(y_test, baseline_pred))


# 8. Define ML models
models = {
    "Ridge Regression": make_pipeline(
        preprocessor,
        Ridge(alpha=1.0)
    ),

    "Lasso Regression": make_pipeline(
        preprocessor,
        Lasso(alpha=1.0, max_iter=10000)
    ),

    "Random Forest": make_pipeline(
        preprocessor,
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    )
}


# 9. Train and evaluate models
results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "R2 Score": r2
    })

    print(f"\n===== {name} =====")
    print("MAE:", mae)
    print("MSE:", mse)
    print("R2 Score:", r2)


# 10. Compare all models
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))


# 11. Best model according to R2
best_model_name = results_df.iloc[0]["Model"]

print("\nBest model according to R2:", best_model_name)
print("\nNumeric feature correlations with Price:")
print(corr["Price"].sort_values(ascending=False))
categorical_cols = [
    "Brand",
    "Fuel Type",
    "Transmission",
    "Condition",
    "Model"
]

for col in categorical_cols:
    print(f"\nAverage Price by {col}:")
    print(
        df.groupby(col)["Price"]
        .agg(["mean", "median", "count"])
        .sort_values("mean", ascending=False)
    )