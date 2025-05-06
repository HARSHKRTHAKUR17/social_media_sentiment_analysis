#data cleaning
import json
import pandas as pd

cleaned_lines = []
with open("tweets_ndjson.json", "r", encoding="utf-8") as f:
    for line in f:
        try:
            obj = json.loads(line)  # strict JSON parsing
            cleaned_lines.append(obj)
        except json.JSONDecodeError:
            continue  # skip problematic lines

df = pd.DataFrame(cleaned_lines)
print(df.head(102))