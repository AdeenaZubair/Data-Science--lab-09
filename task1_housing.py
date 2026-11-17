
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# Load dataset
homes = pd.read_csv("real_estate.csv")

# Convert distance into units of 100 meters
homes["distance_100m"] = (
    homes["X3 distance to the nearest MRT station"] / 100
)

# Select input and target
X = homes[["distance_100m"]]
y = homes["Y house price of unit area"]

# Fit Linear Regression
model = LinearRegression()
model.fit(X, y)

# Print slope and intercept
print("TASK 1: HOUSING PRICE PREDICTION")
print("--------------------------------")

print("Number of transactions:", len(homes))
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Predict price at 500 meters
new_house = pd.DataFrame({
    "distance_100m": [5]
})

prediction = model.predict(new_house)

print("\nPredicted price at 500 meters:", prediction[0])

# Explain slope
print("\nExplanation:")
print(
    f"For every additional 100 meters from the MRT station, "
    f"the predicted house price changes by {model.coef_[0]:.2f} "
    f"price units on average."
)

if model.coef_[0] < 0:
    print("The negative slope indicates that prices decrease as distance increases.")
else:
    print("The positive slope indicates that prices increase as distance increases.")

# Plot observations and fitted line
plt.figure(figsize=(9, 6))

plt.scatter(
    X["distance_100m"],
    y,
    alpha=0.6,
    label="Actual observations"
)

# Generate fitted line
x_line = np.linspace(
    X["distance_100m"].min(),
    X["distance_100m"].max(),
    200
)

x_line_df = pd.DataFrame({
    "distance_100m": x_line
})

y_line = model.predict(x_line_df)

plt.plot(
    x_line,
    y_line,
    color="red",
    linewidth=2,
    label="Fitted regression line"
)

plt.xlabel("Distance from MRT station (100 meters)")
plt.ylabel("House price per unit area")
plt.title("Housing Price Prediction Using Linear Regression")
plt.legend()
plt.grid(True)

plt.savefig("task1_housing.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nPlot saved as task1_housing.png")

# Save numerical results and explanation
with open("lab9_notes.txt", "w") as file:
    file.write("WEEK 09 - DATA SCIENCE LAB RESULTS\n")
    file.write("==================================\n\n")

    file.write("TASK 1: ESTIMATE HOUSING PRICES NEAR TRANSIT\n")
    file.write(f"Number of transactions: {len(homes)}\n")
    file.write(f"Slope: {model.coef_[0]:.4f}\n")
    file.write(f"Intercept: {model.intercept_:.4f}\n")
    file.write(f"Predicted price at 500 meters: {prediction[0]:.4f}\n")

    file.write(
        f"\nExplanation: For every additional 100 meters, "
        f"the predicted price changes by {model.coef_[0]:.2f} "
        f"price units on average.\n"
    )

print("Task 1 results saved in lab9_notes.txt")
