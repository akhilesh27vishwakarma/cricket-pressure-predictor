
import pandas as pd
import joblib
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier

print("Creating training data...")

# Generate many sample cricket situations
data = []

for rate in [x / 4 for x in range(0, 121)]:
    for wickets in range(0, 11):

        # Rule-based labels
        if rate <= 6 and wickets <= 5:
            pressure = "Low"
        elif rate <= 10 and wickets <= 7:
            pressure = "Medium"
        else:
            pressure = "High"

        data.append([rate, wickets, pressure])

df = pd.DataFrame(
    data,
    columns=["required_run_rate", "wickets_lost", "pressure"]
)

X = df[["required_run_rate", "wickets_lost"]]
y = df["pressure"]

# Train Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Save model in the project folder
model_path = Path(__file__).with_name("pressure_model.pkl")
joblib.dump(model, model_path)

print("Model trained successfully!")
print("Model saved:", model_path)
print("Learned categories:", model.classes_)

# Test predictions
test = pd.DataFrame({
    "required_run_rate": [2, 8, 16],
    "wickets_lost": [2, 4, 7]
})

print("\nTest predictions:")
print(model.predict(test))
