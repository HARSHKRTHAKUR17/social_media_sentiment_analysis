import sys
import json
import re
import string

# Ensure UTF-8 encoding
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r'@\w+|#', '', text)  # Remove mentions and hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text.lower().split()  # Return list of words

# Read input line by line for NDJSON
for line in sys.stdin:
    try:
        tweet = json.loads(line.strip())  # Each line is a separate JSON object
        text = tweet.get("text", "").strip()
        if text:
            words = clean_text(text)
            for word in words:
                print(f"{word}\t1")
    except json.JSONDecodeError:
        continue  # Skip malformed JSON lines
