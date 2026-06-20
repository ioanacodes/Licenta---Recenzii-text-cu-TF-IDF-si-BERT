import kagglehub
import pandas as pd
import os

# Descarcă dataset-ul
path = kagglehub.dataset_download("riyosha/letterboxd-movie-reviews-90000")
print("Fișierele sunt salvate în:", path)

# verificăm ce fișiere sunt în folder
print("Conținut folder:", os.listdir(path))

# folosim numele corect al fișierului
file_name = "letterboxd_250movie_reviews.csv"
file_path = os.path.join(path, file_name)

try:
    # citim CSV-ul, sărim peste rândurile corupte
    df = pd.read_csv(file_path, encoding="latin1", on_bad_lines="skip")

    # salvăm în UTF-8
    new_file = file_name.replace(".csv", "-utf8.csv")
    new_path = os.path.join(path, new_file)
    df.to_csv(new_path, index=False, encoding="utf-8")
    print(f"Salvat: {new_file}")

    # detectăm automat coloana cu numele filmului
    possible_cols = ["movie_name", "title", "movie", "film"]
    movie_col = next((c for c in df.columns if c.lower() in possible_cols), df.columns[0])

    # afișăm statistici
    print(f"Număr total review-uri: {len(df)}")
    print(f"Număr filme distincte ({movie_col}): {df[movie_col].nunique()}")
    print(f"Primele 10 filme: {df[movie_col].unique()[:10]}")

except Exception as e:
    print(f"Eroare la {file_name}: {e}")
