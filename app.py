import os
import json
import traceback
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import config
from ml.nlp_pipeline import NLPPipeline
from ml.data_processor import DataProcessor
from ml.model_trainer import ModelTrainer
from ml.predictor import Predictor

app = Flask(__name__, template_folder=config.TEMPLATES_DIR)
CORS(app)

# Global singletons
nlp_pipeline = NLPPipeline()
data_processor = DataProcessor()
predictor = Predictor(nlp_pipeline=nlp_pipeline)

# In-memory training lock and cached state
_is_training = False

@app.route("/")
def index():
    """Serves the primary cybersecurity dashboard application."""
    return render_template("index.html")

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "Spam Mail Detection API"})

@app.route("/api/status", methods=["GET"])
def status():
    """Returns the operational status of the ML pipeline and active model."""
    is_ready = predictor.is_ready()
    eval_data = {}
    if os.path.exists(config.EVALUATION_PATH):
        try:
            with open(config.EVALUATION_PATH, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
        except Exception:
            pass

    return jsonify({
        "trained": is_ready,
        "is_training": _is_training,
        "best_model": eval_data.get("best_model_name", "None" if not is_ready else "Trained Model"),
        "best_model_stats": eval_data.get("best_model_stats", {}),
        "timestamp": eval_data.get("timestamp", None),
        "vocabulary_size": eval_data.get("vocabulary_size", 0)
    })

@app.route("/api/stats", methods=["GET"])
def stats():
    """Returns dataset summary metrics and distributions for visual analytics."""
    try:
        dataset_stats = data_processor.get_stats()
        return jsonify({
            "success": True,
            "stats": dataset_stats
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/results", methods=["GET"])
def results():
    """Returns cached multi-model evaluation leaderboard and confusion matrices."""
    if os.path.exists(config.EVALUATION_PATH):
        try:
            with open(config.EVALUATION_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": f"Failed reading evaluation results: {str(e)}"}), 500

    return jsonify({
        "success": False,
        "message": "Models have not been trained yet. Please initiate training via POST /api/train."
    }), 404

@app.route("/api/train", methods=["POST"])
def train():
    """
    Triggers end-to-end dataset ingestion, NLP preprocessing, TF-IDF feature engineering,
    multi-model evaluation (Naive Bayes, Logistic Regression, Linear SVC, Random Forest),
    and artifact persistence.
    """
    global _is_training, predictor
    if _is_training:
        return jsonify({
            "success": False,
            "message": "Training is already in progress. Please wait."
        }), 409

    try:
        _is_training = True
        trainer = ModelTrainer(nlp_pipeline=nlp_pipeline, data_processor=data_processor)
        results = trainer.run_pipeline()

        # Reload predictor with newly saved artifacts
        predictor.load_artifacts()

        # Also ensure stats are refreshed
        dataset_stats = data_processor.get_stats()

        return jsonify({
            "success": True,
            "results": results,
            "best_model_name": results["best_model_name"],
            "stats": dataset_stats,
            "duration_seconds": results["total_duration_sec"]
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Training failed: {str(e)}"
        }), 500
    finally:
        _is_training = False

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Inference endpoint: evaluates an input message, applies NLP transformation,
    and returns verdict, confidence %, risk level, and step-by-step breakdown.
    """
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")

        prediction = predictor.predict(text)
        if "error" in prediction and prediction.get("status_code") == 503:
            return jsonify(prediction), 503

        return jsonify(prediction)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting SpamShield AI Server on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
