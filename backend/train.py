import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os

# Load dataset
df = pd.read_csv("fake_job_postings.csv")

# Combine all text columns into one
df["text"] = (
    df["title"].fillna("") + " " +
    df["company_profile"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["requirements"].fillna("")
)

# Features & Label
X = df["text"]
y = df["fraudulent"]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_vec = vectorizer.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Train Model with class_weight balanced to fix fake job bias
print("⏳ Training improved model please wait...")
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
print("✅ Training done!")
print(classification_report(y_test, model.predict(X_test)))

# Save model + vectorizer
os.makedirs("../model", exist_ok=True)

with open("../model/fake_job_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("../model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Improved model saved successfully!")