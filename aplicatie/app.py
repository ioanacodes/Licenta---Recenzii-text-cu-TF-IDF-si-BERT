import os
import joblib
import numpy as np
import urllib.request
import urllib.parse
import json
from flask import Flask, request, jsonify, render_template_string
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
SEARCH_DIRS = [BASE, os.path.dirname(BASE)]

def find_file(name, candidates=None):
    names_to_try = [name] + (candidates or [])
    for search_dir in SEARCH_DIRS:
        for n in names_to_try:
            exact = os.path.join(search_dir, n)
            if os.path.exists(exact):
                return exact
        for n in names_to_try:
            stem, ext = os.path.splitext(n)
            for f in os.listdir(search_dir):
                fs, fe = os.path.splitext(f)
                if fe == ext and fs.startswith(stem):
                    return os.path.join(search_dir, f)
    raise FileNotFoundError(
        f"Cannot find '{name}' (tried {names_to_try}) in {SEARCH_DIRS}"
    )

nb_model    = joblib.load(find_file("best_model_nb.pkl", candidates=["best_model_final.pkl"]))
tfidf_rec   = joblib.load(find_file("tfidf_vectorizer_recommender.pkl", candidates=["tfidf_final.pkl"]))
film_matrix = np.load(find_file("film_matrix.npy"))
film_names  = joblib.load(find_file("film_names.pkl"))
label_enc   = joblib.load(find_file("label_encoder.pkl", candidates=["label_encoder_final.pkl"]))

TMDB_KEY  = "8265bd1679663a7ea12ac168da84d2e8" 
TMDB_IMG  = "https://image.tmdb.org/t/p/w300"

def fetch_poster(title: str) -> str:
    try:
        q   = urllib.parse.quote(title)
        url = f"https://api.themoviedb.org/3/search/movie?query={q}&api_key={TMDB_KEY}"
        req = urllib.request.Request(url, headers={"User-Agent": "GuessTheMovie/1.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
        results = data.get("results", [])
        if not results:
            return ""
        path = results[0].get("poster_path", "")
        return (TMDB_IMG + path) if path else ""
    except Exception:
        return ""

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>CineMatch</title>
  <meta name="description" content="Write a movie review and find out which film it belongs to." />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,900;1,400&display=swap" rel="stylesheet" />
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:       #0d0d0d;
      --surface:  #141414;
      --card:     #1a1a1a;
      --card2:    #1f1f1f;
      --border:   #2e2e2e;
      --red:      #e50914;
      --red-dk:   #c1070f;
      --red-glow: rgba(229,9,20,0.22);
      --text:     #efefef;
      --muted:    #888;
      --dim:      #555;
    }

    html { scroll-behavior: smooth; }

    body {
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    header {
      width: 100%;
      padding: 22px 40px;
      display: flex;
      align-items: center;
      gap: 12px;
      border-bottom: 1px solid var(--border);
    }

    .logo {
      display: flex;
      align-items: center;
      gap: 10px;
      text-decoration: none;
    }

    .logo-icon {
      width: 36px; height: 36px;
      background: var(--red);
      border-radius: 7px;
      display: flex; align-items: center; justify-content: center;
      font-size: 18px;
      flex-shrink: 0;
    }

    .logo-name {
      font-size: 1.35rem;
      font-weight: 800;
      color: var(--text);
      letter-spacing: -0.4px;
    }
    .logo-name span { color: var(--red); }

    main {
      width: 100%;
      max-width: 720px;
      padding: 52px 20px 80px;
      display: flex;
      flex-direction: column;
      gap: 28px;
    }

    .hero { text-align: center; }
    .hero h1 {
      font-size: clamp(1.7rem, 4vw, 2.4rem);
      font-weight: 700;
      line-height: 1.2;
      margin-bottom: 10px;
      letter-spacing: -0.5px;
    }
    .hero h1 em {
      font-style: italic;
      color: var(--red);
    }
    .hero p {
      color: var(--muted);
      font-size: 0.94rem;
    }

    .input-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    .input-card:focus-within {
      border-color: rgba(229,9,20,0.45);
      box-shadow: 0 0 0 3px var(--red-glow);
    }

    .field-label {
      font-size: 0.73rem;
      font-weight: 600;
      letter-spacing: 1.3px;
      text-transform: uppercase;
      color: var(--dim);
    }

    textarea#review {
      width: 100%;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 9px;
      color: var(--text);
      font-family: 'Inter', sans-serif;
      font-size: 0.97rem;
      line-height: 1.75;
      padding: 14px 16px;
      resize: vertical;
      min-height: 150px;
      outline: none;
      transition: border-color 0.2s;
    }
    textarea#review::placeholder { color: #3a3a3a; }
    textarea#review:focus { border-color: rgba(229,9,20,0.5); }

    .input-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    #char-count {
      font-size: 0.73rem;
      color: var(--dim);
    }

    #submit-btn {
      display: flex;
      align-items: center;
      gap: 8px;
      background: var(--red);
      color: #fff;
      border: none;
      border-radius: 9px;
      padding: 12px 28px;
      font-family: 'Inter', sans-serif;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
      letter-spacing: 0.2px;
      transition: background 0.15s, transform 0.12s, box-shadow 0.15s;
      box-shadow: 0 3px 16px var(--red-glow);
    }
    #submit-btn:hover   { background: var(--red-dk); transform: translateY(-1px); box-shadow: 0 5px 22px rgba(229,9,20,0.45); }
    #submit-btn:active  { transform: translateY(0); }
    #submit-btn:disabled { opacity: 0.55; cursor: default; transform: none; box-shadow: none; }

    .spinner {
      display: none;
      width: 15px; height: 15px;
      border: 2px solid rgba(255,255,255,0.3);
      border-top-color: #fff;
      border-radius: 50%;
      animation: spin 0.65s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── RESULTS ── */
    #results {
      display: none;
      flex-direction: column;
      gap: 18px;
      animation: fadeUp 0.35s ease both;
    }
    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(14px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    /* Main match card */
    .match-card {
      background: var(--card);
      border: 1px solid rgba(229,9,20,0.3);
      border-radius: 14px;
      overflow: hidden;
      display: flex;
      box-shadow: 0 2px 24px rgba(0,0,0,0.5), 0 0 0 1px rgba(229,9,20,0.1);
    }

    .match-poster {
      width: 110px;
      min-height: 160px;
      flex-shrink: 0;
      background: #111;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      position: relative;
    }
    .match-poster img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    .match-poster .no-poster {
      width: 100%;
      min-height: 160px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 8px;
      background: #111;
    }
    .match-poster .no-poster svg { opacity: 0.18; }
    .match-poster .no-poster p { font-size: 0.6rem; color: var(--dim); text-align: center; padding: 0 6px; letter-spacing: 0.5px; text-transform: uppercase; }

    .match-body {
      padding: 20px 22px;
      flex: 1;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 6px;
    }

    .match-eyebrow {
      font-size: 0.68rem;
      font-weight: 700;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: var(--red);
    }

    .match-title {
      font-size: 1.5rem;
      font-weight: 800;
      line-height: 1.2;
      letter-spacing: -0.3px;
      color: var(--text);
    }

    /* Divider */
    .divider {
      display: flex;
      align-items: center;
      gap: 12px;
      color: var(--dim);
      font-size: 0.7rem;
      font-weight: 600;
      letter-spacing: 1.5px;
      text-transform: uppercase;
    }
    .divider::before, .divider::after {
      content: '';
      flex: 1;
      height: 1px;
      background: var(--border);
    }

    /* Rec items */
    .recs { display: flex; flex-direction: column; gap: 10px; }

    .rec-item {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      display: flex;
      align-items: center;
      gap: 0;
      overflow: hidden;
      transition: border-color 0.15s, transform 0.15s, box-shadow 0.15s;
    }
    .rec-item:hover {
      border-color: rgba(229,9,20,0.35);
      transform: translateX(3px);
      box-shadow: 0 3px 16px rgba(0,0,0,0.3);
    }

    .rec-poster {
      width: 52px;
      height: 76px;
      flex-shrink: 0;
      background: #111;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .rec-poster img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    .rec-poster .no-poster-sm {
      width: 100%;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #111;
    }
    .rec-poster .no-poster-sm svg { opacity: 0.18; }

    .rec-rank {
      width: 36px;
      flex-shrink: 0;
      text-align: center;
      font-size: 0.8rem;
      font-weight: 700;
      color: var(--red);
    }

    .rec-name {
      flex: 1;
      font-size: 0.96rem;
      font-weight: 500;
      color: var(--text);
      padding: 0 4px;
    }

    .rec-pct {
      flex-shrink: 0;
      font-size: 0.72rem;
      font-weight: 600;
      color: var(--muted);
      background: var(--surface);
      border-left: 1px solid var(--border);
      padding: 0 14px;
      height: 76px;
      display: flex;
      align-items: center;
    }

    /* Error */
    .error-box {
      background: rgba(229,9,20,0.07);
      border: 1px solid rgba(229,9,20,0.25);
      border-radius: 10px;
      padding: 16px 20px;
      color: #ff6b6b;
      font-size: 0.9rem;
    }

    footer {
      margin-top: auto;
      padding: 22px;
      text-align: center;
      color: var(--dim);
      font-size: 0.72rem;
      border-top: 1px solid var(--border);
      width: 100%;
    }
  </style>
</head>
<body>

<header>
  <a class="logo" href="/">
    <div class="logo-icon">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="2" y="2" width="20" height="20" rx="2.5"/>
        <line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/>
        <line x1="2" y1="12" x2="22" y2="12"/>
        <line x1="2" y1="7" x2="7" y2="7"/><line x1="17" y1="7" x2="22" y2="7"/>
        <line x1="2" y1="17" x2="7" y2="17"/><line x1="17" y1="17" x2="22" y2="17"/>
      </svg>
    </div>
    <div class="logo-name">Cine<span>Match</span></div>
  </a>
</header>

<main>
  <div class="hero">
    <h1>Write a review.<br/><em>I will try to guess the film :)</em></h1>
  </div>

  <div class="input-card">
    <label class="field-label" for="review">Your review</label>
    <textarea
      id="review"
      placeholder="e.g. A slow-burn psychological thriller set in a cold, remote cabin. The tension builds quietly until the final act completely recontextualizes everything you thought you knew..."
    ></textarea>
    <div class="input-footer">
      <span id="char-count">0 characters</span>
      <button id="submit-btn" onclick="guess()">
        <div class="spinner" id="spinner"></div>
        <span id="btn-label">Guess the movie</span>
      </button>
    </div>
  </div>

  <div id="results"></div>
</main>

<script>
  const textarea  = document.getElementById('review');
  const charCount = document.getElementById('char-count');
  const results   = document.getElementById('results');
  const btn       = document.getElementById('submit-btn');
  const spinner   = document.getElementById('spinner');
  const btnLabel  = document.getElementById('btn-label');

  textarea.addEventListener('input', () => {
    charCount.textContent = textarea.value.length + ' characters';
  });

  async function guess() {
    const text = textarea.value.trim();
    if (!text) { textarea.focus(); return; }

    btn.disabled = true;
    spinner.style.display = 'block';
    btnLabel.textContent = 'Thinking...';
    results.style.display = 'none';

    try {
      const resp = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review: text })
      });
      const data = await resp.json();
      if (data.error) showError(data.error);
      else            showResults(data);
    } catch(e) {
      showError('Something went wrong. Please try again.');
    } finally {
      btn.disabled = false;
      spinner.style.display = 'none';
      btnLabel.textContent = 'Guess the movie';
    }
  }

  const filmSvg = `<svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="20" rx="2.5"/><line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="2" y1="7" x2="7" y2="7"/><line x1="17" y1="7" x2="22" y2="7"/><line x1="2" y1="17" x2="7" y2="17"/><line x1="17" y1="17" x2="22" y2="17"/></svg>`;
  const filmSvgSm = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="20" rx="2.5"/><line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="2" y1="7" x2="7" y2="7"/><line x1="17" y1="7" x2="22" y2="7"/><line x1="2" y1="17" x2="7" y2="17"/><line x1="17" y1="17" x2="22" y2="17"/></svg>`;

  function posterImg(url, alt) {
    if (url) {
      return `<img src="${url}" alt="${escHtml(alt)}" loading="lazy" />`;
    }
    return `<div class="no-poster">${filmSvg}<p>No poster</p></div>`;
  }

  function posterSmall(url, alt) {
    if (url) {
      return `<img src="${url}" alt="${escHtml(alt)}" loading="lazy" />`;
    }
    return `<div class="no-poster-sm">${filmSvgSm}</div>`;
  }

  function showResults(data) {
    const recsHtml = data.recommendations.map((r, i) => `
      <div class="rec-item">
        <div class="rec-poster">${posterSmall(r.poster, r.film)}</div>
        <div class="rec-rank">${i + 1}</div>
        <div class="rec-name">${escHtml(r.film)}</div>
        <div class="rec-pct">${(r.score * 100).toFixed(0)}%</div>
      </div>
    `).join('');

    results.innerHTML = `
      <div class="match-card">
        <div class="match-poster">${posterImg(data.poster, data.predicted_movie)}</div>
        <div class="match-body">
          <div class="match-eyebrow">Best match</div>
          <div class="match-title">${escHtml(data.predicted_movie)}</div>
        </div>
      </div>

      <div class="divider">You might also like</div>

      <div class="recs">${recsHtml}</div>
    `;

    results.style.display = 'flex';
    results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function showError(msg) {
    results.innerHTML = `<div class="error-box">${escHtml(msg)}</div>`;
    results.style.display = 'flex';
  }

  function escHtml(s) {
    return String(s)
      .replace(/&/g,'&amp;')
      .replace(/</g,'&lt;')
      .replace(/>/g,'&gt;')
      .replace(/"/g,'&quot;');
  }

  textarea.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') guess();
  });
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    data   = request.get_json(force=True)
    review = data.get("review", "").strip()
    if not review:
        return jsonify({"error": "Review text is empty."}), 400

    try:
        from sklearn.pipeline import Pipeline as _Pipeline
        if isinstance(nb_model, _Pipeline):
            pred_idx = nb_model.predict([review])[0]
        else:
            pred_idx = nb_model.predict(tfidf_rec.transform([review]))[0]
        movie = label_enc.inverse_transform([pred_idx])[0]

        # Recommendations
        vec  = tfidf_rec.transform([review])
        sims = cosine_similarity(vec, film_matrix).flatten()
        top  = sims.argsort()[::-1][:3]

        recs = [
            {
                "film":   str(film_names[i]),
                "score":  float(sims[i]),
                "poster": fetch_poster(str(film_names[i])),
            }
            for i in top
        ]

        return jsonify({
            "predicted_movie": str(movie),
            "poster":          fetch_poster(str(movie)),
            "recommendations": recs,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
