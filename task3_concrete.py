
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    GridSearchCV
)

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

# Load dataset and remove duplicate rows
concrete = pd.read_csv("concrete_strength.csv")

original_rows = len(concrete)

concrete = concrete.drop_duplicates()

print("TASK 3: CONCRETE STRENGTH MODEL")
print("--------------------------------")

print("Original rows:", original_rows)
print("Rows after removing duplicates:", len(concrete))

# Select features and target
target = "Concrete compressive strength"

X = concrete.drop(columns=target)
y = concrete[target]

print("Number of features:", X.shape[1])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# Five shuffled folds
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# ----------------------------------
# PART A: Linear Regression Baseline
# ----------------------------------

baseline_model = LinearRegression()

baseline_scores = cross_val_score(
    baseline_model,
    X_train,
    y_train,
    cv=cv,
    scoring="neg_root_mean_squared_error"
)

baseline_rmse = -baseline_scores

baseline_mean = baseline_rmse.mean()

print("\nLinear Regression Baseline")
print("Fold RMSE values:", baseline_rmse)
print("Mean baseline CV RMSE:", baseline_mean)

# ----------------------------------
# PART B: Polynomial + Scaling + Ridge
# ----------------------------------

pipeline = Pipeline([
    (
        "polynomial",
        PolynomialFeatures(include_bias=False)
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "ridge",
        Ridge()
    )
])

# Search degrees and alpha values
param_grid = {
    "polynomial__degree": [1, 2, 3],
    "ridge__alpha": [0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Mean CV RMSE:")
best_cv_rmse = -grid_search.best_score_
print(best_cv_rmse)

# Display three highest-ranked settings
results = pd.DataFrame(grid_search.cv_results_)

top_three = results.sort_values(
    by="rank_test_score"
).head(3)

print("\nTop Three Model Settings:")
print(
    top_three[
        [
            "rank_test_score",
            "param_polynomial__degree",
            "param_ridge__alpha",
            "mean_test_score"
        ]
    ].to_string(index=False)
)

# ----------------------------------
# PART C: Test the Best Model
# ----------------------------------

best_model = grid_search.best_estimator_

predictions = best_model.predict(X_test)

test_rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

test_mae = mean_absolute_error(
    y_test,
    predictions
)

print("\nTest Results:")
print("Test RMSE:", test_rmse)
print("Test MAE:", test_mae)

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.7,
    label="Predictions"
)

# Reference line y = x
minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    linewidth=2,
    label="Perfect prediction (y = x)"
)

plt.xlabel("Actual Concrete Strength (MPa)")
plt.ylabel("Predicted Concrete Strength (MPa)")

plt.title("Actual vs Predicted Concrete Strength")

plt.legend()
plt.grid(True)

plt.savefig(
    "task3_concrete.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nPlot saved as task3_concrete.png")

# ----------------------------------
# PART D: Compare Models and Explain
# ----------------------------------

print("\nModel Comparison:")

print("Baseline mean CV RMSE:", baseline_mean)
print("Tuned mean CV RMSE:", best_cv_rmse)

improvement = baseline_mean - best_cv_rmse

print("Improvement in mean CV RMSE:", improvement)

print("\nDegree-2 interaction example:")

# Find a degree-2 interaction term from the feature names
poly = best_model.named_steps["polynomial"]

feature_names = poly.get_feature_names_out(X.columns)

interaction_terms = [
    name for name in feature_names
    if " " in name
]

if interaction_terms:
    print("Example interaction term:", interaction_terms[0])
else:
    print("No interaction term found.")

print(
    "An interaction term represents the combined effect "
    "of two input features on concrete strength."
)

print(
    "\nScaling explanation: StandardScaler is inside the pipeline "
    "so it is fitted separately on each training fold. "
    "This prevents information leakage from validation data."
)

# ----------------------------------
# SAVE RESULTS IN lab9_notes.txt
# ----------------------------------

with open("lab9_notes.txt", "a") as file:

    file.write("\n\nTASK 3: TUNE A CONCRETE STRENGTH MODEL\n")
    file.write("=======================================\n")

    file.write(f"Original rows: {original_rows}\n")
    file.write(f"Rows after removing duplicates: {len(concrete)}\n")

    file.write(f"Training rows: {len(X_train)}\n")
    file.write(f"Testing rows: {len(X_test)}\n")

    file.write("\nBaseline Linear Regression:\n")
    file.write(f"Mean CV RMSE: {baseline_mean:.4f} MPa\n")

    file.write("\nTop Three Settings:\n")

    for _, row in top_three.iterrows():
        file.write(
            f"Rank: {row['rank_test_score']}, "
            f"Degree: {row['param_polynomial__degree']}, "
            f"Alpha: {row['param_ridge__alpha']}, "
            f"Mean CV RMSE: {-row['mean_test_score']:.4f} MPa\n"
        )

    file.write("\nTuned Model:\n")
    file.write(f"Best parameters: {grid_search.best_params_}\n")
    file.write(f"Mean CV RMSE: {best_cv_rmse:.4f} MPa\n")

    file.write("\nTest Performance:\n")
    file.write(f"Test RMSE: {test_rmse:.4f} MPa\n")
    file.write(f"Test MAE: {test_mae:.4f} MPa\n")

    file.write("\nModel Comparison:\n")
    file.write(f"Baseline mean CV RMSE: {baseline_mean:.4f}\n")
    file.write(f"Tuned mean CV RMSE: {best_cv_rmse:.4f}\n")

    file.write(
        "\nExplanation: The tuned model is better when its mean "
        "CV RMSE is lower than the baseline model.\n"
    )

    file.write(
        "\nDegree-2 interaction: A product of two different "
        "input features represents an interaction term.\n"
    )

    file.write(
        "\nScaling belongs inside the pipeline because it must be "
        "fitted only on the training portion of each CV fold. "
        "This prevents data leakage.\n"
    )

print("\nTask 3 results added to lab9_notes.txt")
