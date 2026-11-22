import os
import json
import math
import joblib
import numpy as np

import config
from .nlp_pipeline import NLPPipeline

class Predictor:
    """
    Inference service for real-time spam detection.
    Loads persisted vectorizer and highest-performing classifier.
    Computes class probabilities / sigmoid-scaled decision margins.
    Handles all edge cases gracefully.
    """

    def __init__(self, models_dir=config.MODELS_DIR, nlp_pipeline=None):
        self.models_dir = models_dir
        self.nlp = nlp_pipeline or NLPPipeline()
        self.model = None
        self.vectorizer = None
        self.model_info = {}
        self.load_artifacts()

    def load_artifacts(self) -> bool:
        """Loads saved best_model.pkl, vectorizer.pkl and metadata."""
        try:
            if os.path.exists(config.BEST_MODEL_PATH) and os.path.exists(config.VECTORIZER_PATH):
                self.model = joblib.load(config.BEST_MODEL_PATH)
                self.vectorizer = joblib.load(config.VECTORIZER_PATH)

                if os.path.exists(config.EVALUATION_PATH):
                    with open(config.EVALUATION_PATH, "r", encoding="utf-8") as f:
                        self.model_info = json.load(f)
                return True
        except Exception as e:
            print(f"[Predictor] Error loading model artifacts: {e}")
        return False

    def is_ready(self) -> bool:
        """Returns True if model and vectorizer are loaded and ready."""
        return self.model is not None and self.vectorizer is not None

    def predict(self, text: str) -> dict:
        """
        Analyzes a message and produces detailed classification results.
        Gracefully handles blank inputs, symbols, and extremely long text.
        """
        if not self.is_ready():
            success = self.load_artifacts()
            if not success:
                return {
                    "error": "Model has not been trained yet. Please train the model first.",
                    "status_code": 503
                }

        # 1. Edge Case: Empty or whitespace-only input
        if text is None or not str(text).strip():
            return {
                "verdict": "EMPTY_INPUT",
                "label": "Invalid Input",
                "is_spam": False,
                "confidence": 0.0,
                "confidence_percent": "0.0%",
                "risk_level": "None",
                "raw_text": "",
                "cleaned_text": "",
                "nlp_steps": {},
                "char_count": 0,
                "word_count": 0,
                "warning": "Message is empty. Please enter some text to analyze."
            }

        raw_str = str(text)
        char_count = len(raw_str)
        word_count = len(raw_str.split())

        # 2. NLP Pipeline Cleaning & Step Extraction
        nlp_steps = self.nlp.get_cleaning_steps(raw_str)
        cleaned_text = nlp_steps["final_cleaned"]

        # 3. Edge Case: Text with only symbols/numbers that yield 0 tokens
        if not cleaned_text.strip():
            # If the raw text contains known spam symbols like lots of '$' or '£' or URL fragments
            has_currency = any(sym in raw_str for sym in ['$', '£', '€', '¥'])
            has_urgent_chars = raw_str.count('!') >= 3
            if has_currency and has_urgent_chars:
                return {
                    "verdict": "SPAM",
                    "label": "SPAM",
                    "is_spam": True,
                    "confidence": 75.0,
                    "confidence_percent": "75.0%",
                    "risk_level": "Elevated Suspicion",
                    "raw_text": raw_str,
                    "cleaned_text": "(Non-alphabetic spam indicators detected)",
                    "nlp_steps": nlp_steps,
                    "char_count": char_count,
                    "word_count": word_count,
                    "suspicious_terms": ["High punctuation density", "Currency symbols without text"]
                }
            return {
                "verdict": "SAFE",
                "label": "HAM / Safe",
                "is_spam": False,
                "confidence": 99.0,
                "confidence_percent": "99.0%",
                "risk_level": "Minimal",
                "raw_text": raw_str,
                "cleaned_text": "(No standard words to evaluate)",
                "nlp_steps": nlp_steps,
                "char_count": char_count,
                "word_count": word_count,
                "warning": "Input contained only non-alphabetic characters or stopwords."
            }

        # 4. Feature Extraction using strictly fitted TF-IDF
        X_vec = self.vectorizer.transform([cleaned_text])

        # 5. Prediction & Confidence Estimation
        pred_label = int(self.model.predict(X_vec)[0])

        # Confidence Calculation
        if hasattr(self.model, "predict_proba"):
            # MultinomialNB, LogisticRegression, RandomForest
            probas = self.model.predict_proba(X_vec)[0]
            spam_prob = float(probas[1])
            ham_prob = float(probas[0])
            confidence = (spam_prob if pred_label == 1 else ham_prob) * 100
        elif hasattr(self.model, "decision_function"):
            # LinearSVC decision margin with Sigmoid calibration
            decision_score = float(self.model.decision_function(X_vec)[0])
            # Sigmoid: 1 / (1 + exp(-decision_score))
            # Clip decision score to avoid math overflow
            clipped_score = max(min(decision_score, 20.0), -20.0)
            spam_prob = 1.0 / (1.0 + math.exp(-clipped_score))
            ham_prob = 1.0 - spam_prob
            confidence = (spam_prob if pred_label == 1 else ham_prob) * 100
        else:
            spam_prob = 1.0 if pred_label == 1 else 0.0
            confidence = 90.0

        confidence = round(confidence, 1)
        spam_prob = round(spam_prob, 4)

        # 6. Extract influential / suspicious terms from vocabulary
        suspicious_terms = self._find_suspicious_terms(nlp_steps["lemmatized_tokens"])

        is_spam = bool(pred_label == 1)
        verdict = "SPAM" if is_spam else "SAFE"
        label_display = "SPAM" if is_spam else "HAM / Safe"

        # Risk level determination
        if is_spam:
            if confidence >= 85:
                risk_level = "High-Threat Phishing / Malicious Spam"
            else:
                risk_level = "Moderate Suspicion / Unsolicited Marketing"
        else:
            if confidence >= 85:
                risk_level = "Verified Safe / Authentic Message"
            else:
                risk_level = "Low Risk / Likely Safe"

        return {
            "verdict": verdict,
            "label": label_display,
            "is_spam": is_spam,
            "confidence": confidence,
            "confidence_percent": f"{confidence}%",
            "spam_probability": spam_prob,
            "ham_probability": round(1.0 - spam_prob, 4),
            "risk_level": risk_level,
            "raw_text": raw_str,
            "cleaned_text": cleaned_text,
            "nlp_steps": nlp_steps,
            "char_count": char_count,
            "word_count": word_count,
            "suspicious_terms": suspicious_terms,
            "model_used": self.model_info.get("best_model_name", type(self.model).__name__)
        }

    def _find_suspicious_terms(self, tokens: list) -> list:
        """Identifies prominent spam trigger words present in the message."""
        known_spam_triggers = {
            'free', 'win', 'winner', 'cash', 'prize', 'claim', 'urgent', 'urgent!',
            'call', 'txt', 'text', 'stop', 'reply', 'opt', 'guarantee', 'jackpot',
            'awarded', 'selected', 'congratulations', 'mobile', 'ringtone', 'loan',
            'credit', 'bank', 'verify', 'account', 'password', 'login', 'security',
            'parcel', 'delivery', 'fee', 'bonus', 'million', 'pound', 'dollars'
        }
        found = []
        for t in tokens:
            if t in known_spam_triggers and t not in found:
                found.append(t)
        return found
