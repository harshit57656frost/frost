# SpamShield AI — Spam Mail Prediction Using Machine Learning & NLP

A cyber-security-themed interactive web application for real-time Spam and Phishing detection powered by Natural Language Processing (NLP) and supervised machine learning with Python, Scikit-Learn, and Flask.

---

## 🛡️ Key Features

### 1. Robust Machine Learning & Data Pipeline
- **Dataset Ingestion & Cleaning**: Ingests the genuine English SMS Spam Collection benchmark dataset (5,572 messages). Automatically detects encodings (`utf-8`, `latin-1`), drops nulls/empty entries, eliminates exact duplicate rows (producing 5,158 unique messages: 4,516 Ham and 642 Spam), computes character lengths, and maps labels into binary targets (`ham` = 0, `spam` = 1).
- **Comprehensive NLP Pipeline**:
  - Lowercase normalization
  - Stripping URLs (`http://`, `https://`, `www.`) and email addresses
  - Non-alphabetic character & punctuation removal
  - Word tokenization
  - Standard English stopword filtering (179 stopwords)
  - Root form reduction (Stemming / Lemmatization)
- **Feature Engineering**: TF-IDF (Term Frequency-Inverse Document Frequency) vectorization with a vocabulary cap of 3,000 features, supporting unigrams and bigrams (`(1, 2)`). Strictly fitted only on training data during the stratified 80:20 split to prevent data leakage.
- **4 Supervised Classification Algorithms Evaluated**:
  1. **Linear Support Vector Machine (Linear SVC)** — *Selected Best Model (Accuracy: 98.45%, Spam F1: 93.60%)*
  2. **Multinomial Naïve Bayes** (Laplace smoothing $\alpha = 0.1$) — *Accuracy: 98.06%, Spam F1: 91.80%*
  3. **Random Forest Classifier** (200 trees) — *Accuracy: 97.48%, Spam F1: 88.79%*
  4. **Logistic Regression** (max iterations = 1000) — *Accuracy: 96.21%, Spam F1: 82.35%*
- **Model Evaluation & Artifact Persistence**: Calculates Accuracy, Precision, Recall, F1-Score (weighted and Spam-class specific), and Confusion Matrices. Automatically selects the winner based on the Spam class F1-Score and saves `best_model.pkl` and `vectorizer.pkl`.

### 2. Modern Cyber-Security Web Dashboard
- **Live Message Scanner**:
  - Real-time input box with character & word counter
  - 4 quick-test sample presets (2 Spam/Phishing alerts, 2 normal everyday Ham messages)
  - High-contrast verdict badge (`SPAM` in crimson alert glow vs. `HAM / Safe` in emerald glow)
  - Animated confidence percentage bar (calibrated via class probabilities or sigmoid-scaled decision margins)
  - Threat level categorization and high-risk trigger keyword detection
  - Expandable NLP transformation accordion showing step-by-step preprocessing
  - Edge case resilience (blank inputs, pure symbols, long text)
- **Dataset Insights & Visual Analytics**:
  - 4 metric cards (Raw Messages, Clean Unique, Ham count & %, Spam count & %)
  - Interactive Chart.js donut chart comparing Spam vs. Ham distribution
  - Character length histogram comparing Ham vs. Spam distributions
  - Detailed metrics showing Ham avg length (70.9 chars) vs Spam avg length (137.4 chars)
- **Model Performance & Comparison Lab**:
  - One-click retraining control
  - Side-by-side leaderboard highlighting the winning model
  - Grouped bar chart comparing Accuracy, Weighted F1, and Spam F1 across all 4 models
  - Interactive 2x2 Confusion Matrix heatmap (True Positives, True Negatives, False Positives, False Negatives) with a model switcher

---

## 📁 Project Structure

```
spam mail/
├── app.py                    # Flask application and REST API endpoints
├── run.py                    # One-click launcher (starts server + opens browser)
├── train_cli.py              # CLI training and evaluation verification script
├── test_app.py               # Comprehensive automated test suite
├── config.py                 # Configuration parameters & hyperparameters
├── requirements.txt          # Python dependencies
├── ml/
│   ├── __init__.py
│   ├── nlp_pipeline.py       # Modular NLP text cleaning and tokenization
│   ├── data_processor.py     # Ingestion, cleaning, dedup, and stats calculation
│   ├── model_trainer.py      # TF-IDF + 4 classifiers training, selection & metrics
│   └── predictor.py          # Real-time inference engine and confidence calculation
├── data/
│   ├── sms_spam_collection.tsv # Genuine 5,572 SMS Spam Collection dataset
│   └── dataset_stats.json    # Computed dataset statistics & distribution buckets
├── models/
│   ├── best_model.pkl        # Persisted winning classifier (Linear SVC)
│   ├── vectorizer.pkl        # Persisted TF-IDF vectorizer (3,000 features)
│   └── evaluation_results.json # Multi-model metrics & confusion matrices
└── templates/
    └── index.html            # Dark-mode cybersecurity SPA dashboard
```

---

## 🚀 Running the Application

### 1. Start the Web Dashboard
```bash
python run.py
```
This starts the Flask server on `http://127.0.0.1:5000` and automatically opens your web browser.

Alternatively:
```bash
python app.py
```

### 2. Run the Automated Test Suite
```bash
python test_app.py
```

### 3. Re-train from Command Line (Optional)
```bash
python train_cli.py
```
