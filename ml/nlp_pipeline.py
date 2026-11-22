import re
import string
from nltk.stem import PorterStemmer
import config

# Built-in standard English stopwords list (covers full standard set without requiring external downloads)
STANDARD_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
    'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'did',
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
    "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
    'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
    'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'ur', 'u', 'r'
}

class NLPPipeline:
    """
    High-performance Natural Language Processing pipeline:
    1. Lowercase conversion
    2. URL & email removal
    3. Non-alphabetic character and punctuation stripping
    4. Tokenization (word-level)
    5. Standard English stopword filtering
    6. Root-form reduction (Stemming / Lemmatization)
    """

    def __init__(self):
        self.stopwords = set(STANDARD_STOPWORDS)
        self.stemmer = PorterStemmer()

        # Optional WordNetLemmatizer if available locally
        self.lemmatizer = None
        try:
            from nltk.stem import WordNetLemmatizer
            lem = WordNetLemmatizer()
            _ = lem.lemmatize("testing", pos='v')
            self.lemmatizer = lem
        except Exception:
            self.lemmatizer = None

    def reduce_word_root(self, word: str) -> str:
        """Reduces word to its canonical root form."""
        if self.lemmatizer:
            try:
                return self.lemmatizer.lemmatize(word, pos='v')
            except Exception:
                pass
        return self.stemmer.stem(word)

    def clean_text(self, text: str) -> str:
        """
        Cleans and normalizes raw text for vectorization.
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Lowercase
        cleaned = text.lower()

        # 2. Strip URLs (http, https, www) and email addresses
        cleaned = re.sub(r'https?://\S+|www\.\S+', ' ', cleaned)
        cleaned = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', cleaned)

        # 3. Strip numbers and punctuation (keep only alphabetic characters)
        cleaned = re.sub(r'[^a-z\s]', ' ', cleaned)

        # 4. Tokenization (split on whitespace, keep words with length >= 2)
        tokens = cleaned.split()

        # 5. Filter standard English stopwords
        filtered_tokens = [tok for tok in tokens if tok not in self.stopwords and len(tok) >= 2]

        # 6. Reduce to root forms
        root_tokens = [self.reduce_word_root(tok) for tok in filtered_tokens]

        # 7. Rejoin into space-separated string
        return " ".join(root_tokens)

    def get_cleaning_steps(self, text: str) -> dict:
        """
        Returns all intermediate steps of the NLP transformation pipeline.
        Used by the interactive UI scanner to demonstrate exact text transformations.
        """
        raw = text if isinstance(text, str) else ""
        step_lower = raw.lower()
        step_no_urls = re.sub(r'https?://\S+|www\.\S+', ' ', step_lower)
        step_no_urls = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', step_no_urls)
        step_no_punct_num = re.sub(r'[^a-z\s]', ' ', step_no_urls)
        tokens = [t for t in step_no_punct_num.split() if len(t) >= 2]
        step_tokens = list(tokens)
        step_no_stopwords = [tok for tok in tokens if tok not in self.stopwords]
        step_roots = [self.reduce_word_root(tok) for tok in step_no_stopwords]

        final_cleaned = " ".join(step_roots)

        return {
            "raw": raw,
            "lowercase": step_lower,
            "url_stripped": step_no_urls.strip(),
            "alphabetic_only": re.sub(r'\s+', ' ', step_no_punct_num).strip(),
            "tokens": step_tokens,
            "token_count": len(step_tokens),
            "stopwords_removed": step_no_stopwords,
            "stopwords_removed_count": len(step_no_stopwords),
            "lemmatized_tokens": step_roots,
            "final_cleaned": final_cleaned
        }

    def clean_series(self, series):
        """Vectorized cleaning of pandas series."""
        return series.apply(self.clean_text)
