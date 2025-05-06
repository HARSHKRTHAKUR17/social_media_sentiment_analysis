import matplotlib.pyplot as plt
import pandas as pd
sentiment_counts = df['bert_sentiment'].value_counts()

# Plot the sentiment counts
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])

# Add titles and labels
plt.title('Sentiment Analysis Results', fontsize=16)
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.show()
