# Task 3: NLP with spaCy
# Text Data: Mock Amazon Product Reviews
# Goal: NER for product names, Rule-based Sentiment Analysis.

import spacy
from spacy.tokens import Doc

def main():
    print("-------------------------------------------------")
    print("Task 3: NLP with spaCy (NER & Sentiment)")
    print("-------------------------------------------------")

    # Load English tokenizer, tagger, parser and NER
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading 'en_core_web_sm' model...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # Mock Amazon Reviews
    reviews = [
        "I absolutely love my new Sony headphones! The sound quality is amazing and the bass is deep.",
        "The Samsung Galaxy S21 is a disappointment. The battery life is terrible and it overheats.",
        "Nike running shoes are the best. Very comfortable for long marathons.",
        "I bought a Dell laptop yesterday. It is okay, but the screen brightness is low.",
        "Worst purchase ever. The Fitbit broke after two days. Do not buy."
    ]

    print(f"\nAnalyzing {len(reviews)} reviews...\n")

    # Simple Rule-based Sentiment Analysis
    # We will define a list of positive and negative words.
    # We will check for these words in the review.
    # We will also check for negation (e.g., "not good").
    
    positive_words = {"love", "amazing", "good", "great", "best", "comfortable", "excellent", "deep"}
    negative_words = {"disappointment", "terrible", "bad", "worst", "broke", "low", "hate", "poor"}

    def get_sentiment(doc):
        score = 0
        for token in doc:
            # Check for negation
            is_negated = False
            if token.i > 0 and doc[token.i - 1].dep_ == "neg":
                is_negated = True
            
            word = token.lemma_.lower()
            
            if word in positive_words:
                score += -1 if is_negated else 1
            elif word in negative_words:
                score += 1 if is_negated else -1
        
        if score > 0:
            return "Positive"
        elif score < 0:
            return "Negative"
        else:
            return "Neutral"

    for i, text in enumerate(reviews):
        doc = nlp(text)
        
        print(f"--- Review {i+1} ---")
        print(f"Text: \"{text}\"")
        
        # 1. Named Entity Recognition (NER)
        print("Entities detected:")
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        if entities:
            for ent_text, ent_label in entities:
                print(f"  - {ent_text} ({ent_label})")
        else:
            print("  - None")
            
        # 2. Sentiment Analysis
        sentiment = get_sentiment(doc)
        print(f"Sentiment: {sentiment}")
        print("")

if __name__ == "__main__":
    main()
