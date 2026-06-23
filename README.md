# Licenta---Recenzii-text-cu-TF-IDF-si-BERT

Un proiect de învățare automată care identifică un film pe baza unei recenzii scrise și recomandă titluri similare. Dezvoltat în cadrul lucrării de licență.

---

## Ce face aplicația

Utilizatorul introduce o recenzie de film în interfața web, iar sistemul prezice cărui film îi aparține cel mai probabil recenzia. Pe lângă predicție, aplicația oferă și o listă scurtă de recomandări, ordonate în funcție de similaritatea textuală față de recenzia introdusă.

Proiectul acoperă întregul flux de dezvoltare: colectarea și curățarea datelor, analiza exploratorie, antrenarea și compararea mai multor arhitecturi de clasificare, construirea unui sistem de recomandare bazat pe conținut și integrarea acestora într-o aplicație web realizată cu Flask.

---

## Structura proiectului

```
EVADAREA/
├── aplicatie/
│   ├── app.py                      # Aplicația web Flask
│   ├── best_model_final.pkl        # Clasificatorul Naive Bayes antrenat
│   ├── tfidf_final.pkl             # Vectorizator TF-IDF
│   ├── label_encoder_final.pkl     # Encoder pentru numele filmelor
│   ├── film_matrix.npy             # Matricea profilurilor filmelor pentru recomandări
│   └── film_names.pkl              # Lista numelor de filme asociate matricei
├── cod sursa/
│   └── Licenta_Lese_Ioana.ipynb    # Notebook-ul complet de antrenare
└── date/
    ├── reviews_clean.csv           # Set de date curățat
    ├── reviews_clean_final.csv     # Set de date filtrat final (după 2012)
    └── reviews_english.csv         # Subset doar în limba engleză
```

---

## Setul de date

Datele provin din setul de date **Letterboxd Movie Reviews** disponibil pe Kaggle (https://www.kaggle.com/datasets/riyosha/letterboxd-movie-reviews-90000) și conțin recenzii pentru filmele din clasamentul Letterboxd Top 250.

După procesarea și filtrarea datelor, setul utilizat în proiect conține recenzii publicate începând cu anul 2012, fiind eliminate duplicatele și intrările foarte scurte.

Fiecare înregistrare conține textul unei recenzii și numele filmului corespunzător, acesta reprezentând variabila țintă pentru clasificare.

---

## Arhitecturi de învățare automată

În notebook sunt antrenate și comparate mai multe abordări înainte de alegerea modelului final.

### TF-IDF + Naive Bayes (modelul final)

Textele sunt transformate în reprezentări numerice folosind TF-IDF cu unigrame, un vocabular de 30.000 de termeni, scalare subliniară a frecvenței termenilor și eliminarea cuvintelor comune din limba engleză.

Peste aceste reprezentări este antrenat un clasificator Multinomial Naive Bayes cu netezire de tip alpha.

Această combinație a oferit cel mai bun echilibru între acuratețe și timpul de antrenare pentru problema de clasificare cu 250 de clase.

### TF-IDF + Rețea neuronală (SGD)

A fost utilizată o rețea neuronală simplă cu un singur strat dens și funcție softmax, antrenată prin algoritmul Stochastic Gradient Descent pe caracteristici TF-IDF.

Au fost testate mai multe valori pentru rata de învățare, momentum și configurații Nesterov, folosind vocabulare bazate pe unigrame, bigrame și trigrame.

### Fine-tuning BERT

A fost utilizat modelul `bert-base-uncased` împreună cu un strat de clasificare.

Toate straturile encoderului au fost înghețate, cu excepția ultimului, iar modelul a fost antrenat timp de 3 epoci folosind rate de învățare diferențiate: mai mici pentru encoder și mai mari pentru clasificator.

Această abordare a avut rol explorator și a presupus costuri computaționale ridicate, fiind inclusă în principal pentru comparație.

### Sistem de recomandare bazat pe conținut

Fiecare film este reprezentat printr-un profil TF-IDF obținut prin media tuturor recenziilor asociate și normalizat la lungime unitară.

În etapa de inferență, recenzia introdusă este vectorizată folosind același model TF-IDF, iar similaritatea cosinus este calculată față de toate profilurile filmelor.

Sunt returnate primele trei filme cu cel mai mare scor de similaritate.

---

## Rularea aplicației

Asigură-te că Python este instalat împreună cu pachetele necesare.

### Instalarea dependențelor

```bash
pip install flask scikit-learn numpy joblib
```

### Pornirea serverului

```bash
cd aplicatie
python app.py
```

Deschide apoi browserul și accesează:

```text
http://127.0.0.1:5000
```

Introdu o recenzie în câmpul de text și apasă butonul pentru predicție. Sistemul va afișa filmul prezis și trei recomandări.

---

## Generarea fișierelor modelului

Fișierele modelului antrenat (`.pkl`, `.npy`) și fișierele setului de date (`.csv`) nu sunt incluse în acest repository deoarece depășesc limita de dimensiune impusă de GitHub.

Acestea trebuie generate local prin rularea notebook-ului.

Descarcă mai întâi setul de date de pe Kaggle:

https://www.kaggle.com/datasets/riyosha/letterboxd-movie-reviews-90000

Apoi rulează notebook-ul integral:

```text
cod sursa/Licenta_Lese_Ioana.ipynb
```

Notebook-ul presupune existența fișierelor `reviews_clean.csv` și `reviews_clean_final.csv` în directorul de lucru.

Celulele 37 și 38 salvează toate fișierele necesare ale modelului. Acestea trebuie mutate ulterior în directorul `aplicatie/` înainte de pornirea serverului.

---

## Cerințe

* Python 3.9 sau versiune ulterioară
* scikit-learn
* numpy
* joblib
* flask

### Pentru rularea completă a notebook-ului

* pandas
* matplotlib
* seaborn
* tensorflow (pentru experimentele cu rețele neuronale)
* transformers + torch (pentru experimentele cu BERT)

**Lese Ioana**
Lucrare de licență, 2026
