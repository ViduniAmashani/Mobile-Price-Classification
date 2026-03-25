# app.py
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -----------------------------
# Step 1: Train Models
# -----------------------------
train_df = pd.read_csv("train.csv")

X = train_df.drop("price_range", axis=1)
y = train_df["price_range"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# -----------------------------
# Step 2: UI
# -----------------------------
st.title("📱 Mobile Price Prediction App")

st.markdown("""
### 📊 Price Range Meaning:
- **0 → Low Cost**
- **1 → Medium Cost**
- **2 → High Cost**
- **3 → Very High Cost**
""")

uploaded_file = st.file_uploader("📂 Upload your test.csv", type="csv")

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data (first 5 rows)")
    st.dataframe(test_df.head())

    # Prepare data
    test_ids = test_df["id"] if "id" in test_df.columns else range(len(test_df))
    test_features = test_df.drop("id", axis=1) if "id" in test_df.columns else test_df

    if "price_range" in test_features.columns:
        test_features = test_features.drop("price_range", axis=1)

    # Scale for KNN
    test_scaled = scaler.transform(test_features)

    # Predictions
    pred_knn = knn.predict(test_scaled)
    pred_rf = rf.predict(test_features)

    # Output
    output = pd.DataFrame({
        "id": test_ids,
        "KNN Prediction": pred_knn,
        "Random Forest Prediction": pred_rf
    })

    # -----------------------------
    # Show Predictions Table
    # -----------------------------
    st.subheader("📋 Predictions (first 10 rows)")
    st.dataframe(output.head(10))

    # -----------------------------
    # Summary Counts
    # -----------------------------
    st.subheader("📊 Prediction Summary (Random Forest)")

    counts = pd.Series(pred_rf).value_counts().sort_index()

    for i in range(4):
        st.write(f"Price Range {i}: {counts.get(i, 0)} mobiles")

    # -----------------------------
    # Bar Chart
    # -----------------------------
    st.subheader("📈 Visualization")

    plt.figure()
    counts.plot(kind='bar')
    plt.xlabel("Price Range")
    plt.ylabel("Number of Mobiles")
    plt.title("Distribution of Predicted Price Ranges")

    st.pyplot(plt)

    # -----------------------------
    # Download Button
    # -----------------------------
    csv = output.to_csv(index=False).encode()
    st.download_button(
        label="📥 Download Predictions",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )