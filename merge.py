import pandas as pd

# 1. Load both
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# 2. Add label column
fake['label'] = 0  # 0 = FAKE
true['label'] = 1  # 1 = REAL

# 3. Merge
df = pd.concat([fake, true], ignore_index=True)

# 4. Shuffle (important)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 5. Clean - remove empty text
df = df.dropna(subset=['text'])
df['text'] = df['text'].fillna('')

# 6. Save final
df.to_csv("news_dataset.csv", index=False)

print(f"Fake: {len(fake)}")
print(f"True: {len(true)}")
print(f"Merged Total: {len(df)}")
print(df['label'].value_counts())
print("DONE - news_dataset.csv created")