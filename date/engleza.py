import pandas as pd
import langid

file_path = r"C:\Users\User 14\.cache\kagglehub\datasets\riyosha\letterboxd-movie-reviews-90000\versions\1\letterboxd_250movie_reviews.csv"

# Citește fișierul
df = pd.read_csv(file_path)

# Detectează limba fiecărui review
df["language"] = df["Review"].apply(lambda x: langid.classify(str(x))[0])

# Numără câte review-uri pe fiecare limbă
print(df["language"].value_counts())

# Salvează doar review-urile în engleză (codul 'en')
df_en = df[df["language"] == "en"]

print(f"Total review-uri inițiale: {len(df)}")
print(f"Review-uri în engleză: {len(df_en)}")

# Exportăm subsetul filtrat într-un nou CSV
output_path = r"D:\EVADAREA\date\reviews_english.csv"
df_en.to_csv(output_path, index=False, encoding="utf-8")
print("Fișier salvat la:", output_path)
