#Random Forest(main)
import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

# 🔹 Load CSV data
df_csv = pd.read_csv("bot_detection_data.csv")
print("Original CSV:\n", df_csv.head())

# 🔹 Clean text
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

df_csv['clean_text'] = df_csv['Tweet'].astype(str).apply(clean_text)

# 🔹 TF-IDF Vectorizer with n-grams
vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1,2), stop_words='english')
X = vectorizer.fit_transform(df_csv['clean_text'])
y = df_csv['Bot Label']

# 🔹 Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 🔹 GridSearchCV for Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

# 🔹 Best model evaluation
model = grid.best_estimator_
y_pred = model.predict(X_test)
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

# 🔹 Load and clean delimited JSON
with open("tweets_ndjson.json", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    data = [json.loads(line) for line in lines]

df_json = pd.DataFrame(data)
df_json['clean_text'] = df_json['text'].astype(str).apply(clean_text)

# 🔹 Vectorize and Predict
X_json = vectorizer.transform(df_json['clean_text'])
predictions = model.predict(X_json)
df_json['Prediction'] = predictions

# 🔹 Print ALL tweets with predictions
print("\n🔮 All Tweets with Predictions:\n", df_json[['text', 'Prediction']].to_string(index=False))
