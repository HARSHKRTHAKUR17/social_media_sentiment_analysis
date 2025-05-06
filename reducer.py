import sys
from collections import defaultdict

# Ensure UTF-8 encoding
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

word_counts = defaultdict(int)

# Process input
for line in sys.stdin:
    try:
        word, count = line.strip().split("\t")
        word_counts[word] += int(count)
    except ValueError:
        continue  # Ignore incorrect lines

# Sort words by frequency in descending order
sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

# Print output
for word, count in sorted_word_counts:
    print(f"{word}\t{count}")
