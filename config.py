import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

DATASET_PATH = os.path.join(DATA_DIR, "sms_spam_collection.tsv")
EVALUATION_PATH = os.path.join(MODELS_DIR, "evaluation_results.json")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
STATS_PATH = os.path.join(DATA_DIR, "dataset_stats.json")

# ML Hyperparameters & Config
RANDOM_STATE = 42
TEST_SIZE = 0.2
TFIDF_MAX_FEATURES = 3000
TFIDF_NGRAM_RANGE = (1, 2)

# Ensure required directories exist
for directory in [DATA_DIR, MODELS_DIR, NLTK_DATA_DIR, TEMPLATES_DIR]:
    os.makedirs(directory, exist_ok=True)
