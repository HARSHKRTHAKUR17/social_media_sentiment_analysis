#logistic Regression
import pandas as pd
import numpy as np
import re
import string
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 2: Load and Clean Labeled CSV Dataset ---
# Load CSV data for training
df = pd.read_csv('bot_detection_data.csv')

# Print first few rows to ensure correct loading
print("Data Before Cleaning (from CSV):")
print(df[['Tweet', 'Bot Label']].head())

# Check for unique values in 'Bot Label' to identify potential issues
print("\nUnique values in 'Bot Label':", df['Bot Label'].unique())

# Remove any leading/trailing spaces in the 'Tweet' column only
df['Tweet'] = df['Tweet'].str.strip()

# Check for null values and types
print("\nMissing values after cleaning:")
print(df[['Tweet', 'Bot Label']].isnull().sum())

# Drop rows with missing 'Tweet' or 'Bot Label' values
df = df[['Tweet', 'Bot Label']].dropna()

# Convert Bot Label to binary (1 for bot, 0 for human)
df['Bot Label'] = df['Bot Label'].astype(int)

# Check after conversion
print("\nData After Label Conversion (from CSV):")
print(df[['Tweet', 'Bot Label']].head())

# --- Step 3: Tweet Cleaning Function ---
def clean_text(text):
    text = text.lower()  # Lowercase the text
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r"@\w+", '', text)  # Remove mentions
    text = re.sub(r"#\w+", '', text)  # Remove hashtags
    text = re.sub(r"[%s]" % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r"\d+", '', text)  # Remove digits
    return text.strip()

# Apply the clean_text function to the Tweet column
df['clean_text'] = df['Tweet'].apply(clean_text)

# Check the result after cleaning
print("\nData After Cleaning (from CSV):")
print(df[['Tweet', 'clean_text']].head())

# --- Step 4: Split, Vectorize, Train Model ---
# Ensure there are valid rows before continuing
if not df.empty:
    X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['Bot Label'], test_size=0.2, random_state=42)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # --- Step 5: Evaluate Model ---
    y_pred = model.predict(X_test_vec)
    print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

    # --- Step 6: Visualize Confusion Matrix ---
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
else:
    print("DataFrame is empty after cleaning. Please check the 'Tweet' and 'Bot Label' columns.")

# --- Step 7: Load and Clean New Delimited JSON Data for Prediction ---
# Load the delimited JSON file containing new data (where each line is a JSON object)
df_json = pd.read_json('tweets_ndjson.json', lines=True)

# Print first few rows to ensure correct loading
print("\nData Before Cleaning (from Delimited JSON):")
print(df_json[['text']].head())

# Apply the clean_text function to the 'text' column in the JSON data
df_json['clean_text'] = df_json['text'].apply(clean_text)

# Print cleaned data to verify
print("\nData After Cleaning (from Delimited JSON):")
print(df_json[['text', 'clean_text']].head())

# --- Step 8: Predict Using the Trained Model ---
# Vectorize the new cleaned data using the same vectorizer
X_new_vec = vectorizer.transform(df_json['clean_text'])

# Make predictions using the trained model
predictions = model.predict(X_new_vec)

# Add the predictions as a new column in the dataframe
df_json['Prediction'] = predictions

# Print the final data with predictions
print("\nPredictions for New Data (from Delimited JSON):")
print(df_json[['text', 'Prediction']].head())
