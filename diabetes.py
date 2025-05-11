import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load dataset
df = pd.read_csv("diabetes.csv")

# Step 2: Split features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build SVM model
model = SVC(kernel='linear')  # You can use 'rbf', 'poly', or 'sigmoid' too
model.fit(X_train, y_train)

# Step 5: Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Predict for a new sample
new_sample = [[5, 120, 70, 20, 79, 25.0, 0.5, 30]]  # Sample input
prediction = model.predict(new_sample)
print("\nPrediction for new sample:", "Diabetic" if prediction[0] == 1 else "Not Diabetic")
