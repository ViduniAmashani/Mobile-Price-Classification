# main.py
# Mobile Price Classification - KNN & Random Forest

# -----------------------------
# Step 1: Import Libraries
# -----------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Step 2: Load Dataset
# -----------------------------
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print("Training data sample:")
print(train_df.head())
print("\nTesting data sample:")
print(test_df.head())

# -----------------------------
# Step 3: Split Features & Target
# -----------------------------
from sklearn.model_selection import train_test_split

X = train_df.drop("price_range", axis=1)
y = train_df["price_range"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 4: Feature Scaling (for KNN)
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Step 5: Train Models
# -----------------------------
# 5a. K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

# 5b. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# -----------------------------
# Step 6: Evaluate Models
# -----------------------------
print("\n--- KNN Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

print("\n--- Random Forest Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Confusion Matrices
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap="Blues")
plt.title("KNN Confusion Matrix")

plt.subplot(1,2,2)
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap="Greens")
plt.title("Random Forest Confusion Matrix")

plt.show()

# -----------------------------
# Step 7: Compare Accuracy
# -----------------------------
accuracies = {
    "KNN": accuracy_score(y_test, y_pred_knn),
    "Random Forest": accuracy_score(y_test, y_pred_rf)
}

plt.bar(list(accuracies.keys()), list(accuracies.values()), color=["blue", "green"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

# -----------------------------
# Step 8: Predict on real test.csv
# -----------------------------
# Remove 'id' column (not a feature)
test_ids = test_df["id"]
test_features = test_df.drop("id", axis=1)

# If 'price_range' exists, remove it for features and keep for evaluation
if "price_range" in test_features.columns:
    y_test_true = test_features["price_range"]
    test_features = test_features.drop("price_range", axis=1)
else:
    y_test_true = None

# Scale for KNN
test_features_scaled = scaler.transform(test_features)

# Predictions
test_pred_knn = knn.predict(test_features_scaled)
test_pred_rf = rf.predict(test_features)

# Evaluate on test data if true labels exist
if y_test_true is not None:
    from sklearn.metrics import accuracy_score, classification_report
    print("\n--- Evaluation on test.csv ---")
    print("KNN Accuracy:", accuracy_score(y_test_true, test_pred_knn))
    print("Random Forest Accuracy:", accuracy_score(y_test_true, test_pred_rf))
    print("\nRandom Forest Classification Report:\n", classification_report(y_test_true, test_pred_rf))

# Save Random Forest predictions (best model)
output = pd.DataFrame({
    "id": test_ids,
    "price_range": test_pred_rf
})

output.to_csv("predictions.csv", index=False)
print("\nPredictions saved to predictions.csv")