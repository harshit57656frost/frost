import os
import json
import time
from datetime import datetime
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import config
from .nlp_pipeline import NLPPipeline
from .data_processor import DataProcessor

class ModelTrainer:
    """
    Orchestrates the complete NLP + ML pipeline:
    - Text preprocessing
    - Stratified train-test split (80:20, fixed seed)
    - TF-IDF vectorization (max 3000 features, unigrams+bigrams, fit on train only)
    - 4 distinct model training:
        1. Multinomial Naive Bayes (alpha=0.1)
        2. Logistic Regression (max_iter=1000)
        3. Linear Support Vector Machine (LinearSVC)
        4. Random Forest Classifier (200 estimators)
    - Multi-metric evaluation and confusion matrix computation
    - Winning model selection based on Spam Class F1-Score
    - Model & Vectorizer artifact persistence
    """

    def __init__(self, nlp_pipeline=None, data_processor=None):
        self.nlp = nlp_pipeline or NLPPipeline()
        self.processor = data_processor or DataProcessor()
        self.vectorizer = None
        self.models = {}
        self.evaluation_results = {}
        self.best_model_name = ""
        self.best_model = None

    def run_pipeline(self) -> dict:
        """
        Executes end-to-end training and evaluation workflow.
        Returns full results dictionary.
        """
        start_time = time.time()
        print("[ModelTrainer] Starting full ML training pipeline...")

        # 1. Ingest and Clean Dataset
        df = self.processor.load_and_clean()

        # 2. NLP Preprocessing on all unique messages
        print(f"[ModelTrainer] Preprocessing {len(df)} messages through NLP pipeline...")
        df['cleaned_message'] = df['message'].apply(self.nlp.clean_text)

        # Filter out messages that became completely empty after cleaning (if any)
        valid_mask = df['cleaned_message'].str.strip().str.len() > 0
        df_valid = df[valid_mask].copy()

        X = df_valid['cleaned_message']
        y = df_valid['target']

        print(f"[ModelTrainer] Performing Stratified 80:20 split (Random State={config.RANDOM_STATE})...")
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            stratify=y,
            random_state=config.RANDOM_STATE
        )

        # 3. TF-IDF Feature Engineering (strictly fit ONLY on training data to prevent leakage)
        print(f"[ModelTrainer] Fitting TF-IDF Vectorizer (max_features={config.TFIDF_MAX_FEATURES}, ngrams={config.TFIDF_NGRAM_RANGE})...")
        self.vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            sublinear_tf=True
        )

        X_train = self.vectorizer.fit_transform(X_train_raw)
        X_test = self.vectorizer.transform(X_test_raw)

        print(f"[ModelTrainer] Vectorized matrices: Train {X_train.shape}, Test {X_test.shape}")

        # 4. Instantiate the 4 Specified Classifiers
        models_to_train = {
            "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
            "Linear Support Vector Machine": LinearSVC(max_iter=2000, random_state=config.RANDOM_STATE),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=200, random_state=config.RANDOM_STATE, n_jobs=1)
        }

        model_results = []
        trained_models = {}

        # 5. Train & Evaluate Each Algorithm
        for name, model in models_to_train.items():
            t0 = time.time()
            print(f"[ModelTrainer] Training {name}...")
            model.fit(X_train, y_train)
            fit_time = time.time() - t0

            # Predict on unseen test set
            y_pred = model.predict(X_test)

            # Metrics calculation
            acc = float(accuracy_score(y_test, y_pred))
            prec_weighted = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            rec_weighted = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            f1_weighted = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

            # Spam class specific metrics (label=1)
            spam_prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
            spam_rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
            spam_f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = [int(v) for v in cm.ravel()]

            res_entry = {
                "name": name,
                "accuracy": round(acc * 100, 2),
                "precision": round(prec_weighted * 100, 2),
                "recall": round(rec_weighted * 100, 2),
                "f1_score": round(f1_weighted * 100, 2),
                "spam_precision": round(spam_prec * 100, 2),
                "spam_recall": round(spam_rec * 100, 2),
                "spam_f1": round(spam_f1 * 100, 2),
                "confusion_matrix": {
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                    "matrix_2d": [[tn, fp], [fn, tp]]
                },
                "training_time_sec": round(fit_time, 3)
            }

            model_results.append(res_entry)
            trained_models[name] = model

        # 6. Automatic Selection of Highest-Performing Model based on Spam Class F1-Score
        # Sort leaderboard descending by spam_f1, then overall f1_score
        model_results.sort(key=lambda x: (x["spam_f1"], x["f1_score"]), reverse=True)
        best_entry = model_results[0]
        self.best_model_name = best_entry["name"]
        self.best_model = trained_models[self.best_model_name]

        for entry in model_results:
            entry["is_best"] = (entry["name"] == self.best_model_name)

        total_duration = round(time.time() - start_time, 2)
        print(f"[ModelTrainer] Best Model Selected: '{self.best_model_name}' with Spam F1 = {best_entry['spam_f1']}%")

        # 7. Persistence: Save Winning Classifier and TF-IDF Vectorizer
        print(f"[ModelTrainer] Saving winning model and vectorizer to {config.MODELS_DIR}...")
        joblib.dump(self.best_model, config.BEST_MODEL_PATH)
        joblib.dump(self.vectorizer, config.VECTORIZER_PATH)

        self.evaluation_results = {
            "success": True,
            "best_model_name": self.best_model_name,
            "best_model_stats": best_entry,
            "leaderboard": model_results,
            "test_set_size": len(y_test),
            "train_set_size": len(y_train),
            "vocabulary_size": len(self.vectorizer.vocabulary_),
            "total_duration_sec": total_duration,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save evaluation summary to JSON
        with open(config.EVALUATION_PATH, "w", encoding="utf-8") as f:
            json.dump(self.evaluation_results, f, indent=2)

        return self.evaluation_results
