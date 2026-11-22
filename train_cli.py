import sys
import os
import config
from ml.nlp_pipeline import NLPPipeline
from ml.data_processor import DataProcessor
from ml.model_trainer import ModelTrainer
from ml.predictor import Predictor

def main():
    print("=" * 60)
    print("Spam Mail ML Pipeline & Model Training Test")
    print("=" * 60)

    # 1. Initialize NLP
    print("[1] Initializing NLP Pipeline...")
    nlp = NLPPipeline()
    test_text = "Congratulations! You've WON a $1000 cash prize! Visit http://scam.site/claim NOW!"
    cleaned = nlp.clean_text(test_text)
    steps = nlp.get_cleaning_steps(test_text)
    print(f"    Raw:     '{test_text}'")
    print(f"    Cleaned: '{cleaned}'")
    print(f"    Tokens:  {steps['lemmatized_tokens']}")

    # 2. Ingestion & Statistics
    print("\n[2] Ingesting SMS Spam Collection Dataset...")
    processor = DataProcessor()
    df = processor.load_and_clean()
    stats = processor.get_stats()
    print(f"    Total Raw Messages:     {stats['total_raw']}")
    print(f"    Clean Unique Messages:  {stats['total_clean']}")
    print(f"    Ham Messages:           {stats['ham_count']} ({stats['ham_percentage']}%)")
    print(f"    Spam Messages:          {stats['spam_count']} ({stats['spam_percentage']}%)")
    print(f"    Avg Ham Char Length:    {stats['avg_ham_char_length']}")
    print(f"    Avg Spam Char Length:   {stats['avg_spam_char_length']}")

    # 3. Model Training & Evaluation
    print("\n[3] Training 4 Classifiers on Stratified 80:20 Split...")
    trainer = ModelTrainer(nlp_pipeline=nlp, data_processor=processor)
    results = trainer.run_pipeline()

    print(f"\n[4] Evaluation Results (Duration: {results['total_duration_sec']}s):")
    print(f"    Winning Model: {results['best_model_name']}")
    print("-" * 75)
    print(f"{'Model Name':<32} | {'Acc':<6} | {'Prec':<6} | {'Recall':<6} | {'F1':<6} | {'Spam F1':<6}")
    print("-" * 75)
    for m in results["leaderboard"]:
        flag = " [BEST]" if m.get("is_best") else ""
        print(f"{m['name']:<32} | {m['accuracy']:<6.2f} | {m['precision']:<6.2f} | {m['recall']:<6.2f} | {m['f1_score']:<6.2f} | {m['spam_f1']:<6.2f}{flag}")
    print("-" * 75)

    # 4. Predictor Test
    print("\n[5] Testing Inference Service...")
    pred = Predictor(nlp_pipeline=nlp)
    test_samples = [
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt CLAIM to 81010",
        "SECURITY ALERT: Unauthorized login detected on your bank account. Verify at http://bank-update.xyz",
        "Hey are you coming to dinner tonight? Let me know what time works.",
        "The project presentation is ready for tomorrow's standup meeting.",
        "", # empty test
        "???$$$!!!" # symbols test
    ]

    for s in test_samples:
        p = pred.predict(s)
        print(f"\n    Sample: '{s[:50]}...' " if len(s) > 50 else f"\n    Sample: '{s}'")
        print(f"    Verdict: {p['verdict']} ({p['label']}) | Confidence: {p.get('confidence_percent')} | Risk: {p.get('risk_level')}")
        if p.get('suspicious_terms'):
            print(f"    Triggers: {p.get('suspicious_terms')}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
