import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

print("Loading dataset...")
# Load your CSVs - Make sure Fake.csv and True.csv are in same folder
try:
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")
except FileNotFoundError:
    print("ERROR: Put Fake.csv and True.csv in same folder as train.py")
    exit()

fake['label'] = 0 # Fake = 0
true['label'] = 1 # Real = 1

df = pd.concat([fake, true])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df['text'].fillna('').astype(str)
y = df['label']

print(f"Total articles: {len(df)}")
print(f"Fake: {len(fake)}, Real: {len(true)}")

# TF-IDF
print("Vectorizing...")
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=5000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Model
print("Training Logistic Regression...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"\nAccuracy: {acc*100:.2f}%")
print(classification_report(y_test, model.predict(X_test)))

# Save
os.makedirs("model", exist_ok=True)
pickle.dump(vectorizer, open("model/tfidf_vectorizer.pkl","wb"))
pickle.dump(model, open("model/lr_model.pkl","wb"))
print("\nSaved to model/tfidf_vectorizer.pkl and model/lr_model.pkl")