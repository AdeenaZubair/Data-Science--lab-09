
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_score

# Load headerless yacht dataset
yachts = pd.read_csv(
    "yacht_hydrodynamics.data",
    sep=r"\s+",
    header=None
)

# Select sixth and seventh columns
X = yachts[[5]]
y = yachts[6]

# Five shuffled folds
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Model 1: Linear Regression
linear_model = LinearRegression()

linear_scores = cross_val_score(
    linear_model,
    X,
    y,
    cv=cv,
    scoring="neg_root_mean_squared_error"
)

# Convert negative RMSE into positive RMSE
linear_rmse = -linear_scores

# Model 2: Degree-2 Polynomial Regression
polynomial_model = Pipeline([
    ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
    ("linear", LinearRegression())
])

polynomial_scores = cross_val_score(
    polynomial_model,
    X,
    y,
    cv=cv,
    scoring="neg_root_mean_squared_error"
)

polynomial_rmse = -polynomial_scores

# Display results
print("TASK 2: YACHT RESISTANCE MODEL COMPARISON")
print("-----------------------------------------")

print("Number of experiments:", len(yachts))

print("\nLinear Regression:")
for i, score in enumerate(linear_rmse, start=1):
    print(f"Fold {i} RMSE: {score:.4f}")

linear_mean = linear_rmse.mean()
print("Mean Linear RMSE:", linear_mean)

print("\nDegree-2 Polynomial Regression:")
for i, score in enumerate(polynomial_rmse, start=1):
    print(f"Fold {i} RMSE: {score:.4f}")

polynomial_mean = polynomial_rmse.mean()
print("Mean Polynomial RMSE:", polynomial_mean)

# Choose better model
print("\nModel Comparison:")

if polynomial_mean < linear_mean:
    better_model = "Degree-2 Polynomial Regression"
    print("Better model:", better_model)
else:
    better_model = "Linear Regression"
    print("Better model:", better_model)

# Append results to notes
with open("lab9_notes.txt", "a") as file:

    file.write("\n\nTASK 2: COMPARE MODELS FOR YACHT RESISTANCE\n")
    file.write("============================================\n")

    file.write("\nLinear Regression Fold RMSEs:\n")
    for i, score in enumerate(linear_rmse, start=1):
        file.write(f"Fold {i}: {score:.4f}\n")

    file.write(f"Mean Linear RMSE: {linear_mean:.4f}\n")

    file.write("\nDegree-2 Polynomial Regression Fold RMSEs:\n")
    for i, score in enumerate(polynomial_rmse, start=1):
        file.write(f"Fold {i}: {score:.4f}\n")

    file.write(f"Mean Polynomial RMSE: {polynomial_mean:.4f}\n")
    file.write(f"\nBetter model: {better_model}\n")

    file.write(
        "\nExplanation: The model with the lower mean RMSE "
        "provides better average prediction accuracy. "
        "The row-based split may place experiments from the same "
        "hull in different folds.\n"
    )

print("\nTask 2 results added to lab9_notes.txt")
