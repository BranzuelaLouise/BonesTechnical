import pandas as pd
import spacy
from transformers import pipeline

FILLER_WORDS = [
     "um", "uh", "like", "you know", "i mean", "well", "so", "basically",
     "actually", "er", "ah", "hmm", "hm", "you know what i mean", "you know",
     "for sure"
]

model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english" # Prevent clogging the standard output

def main():
    lines = load_transcript("transcript.txt")

    for line in lines:
        sentiment = compute_sentiment(line)
        filler_ratio = compute_filler_ratio(line)
        print(f"Sentiment: {sentiment['label']}, Filler Ratio: {filler_ratio}")

def load_transcript(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines if line.strip()]
    return lines

def compute_sentiment(text):
        sentiment_pipeline = pipeline("sentiment-analysis", model=model)
        if not text or text.isspace():
            return {"label": "NEUTRAL", "score": 0.0}
        result = sentiment_pipeline(text)[0]
        return result

def compute_filler_ratio(text):
    total_words = len(text.split())

    filler_count = 0
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text.lower())

    for token in doc:
         if token.text in FILLER_WORDS:
              filler_count += 1
              print(token.text)
    
    ratio = filler_count / total_words
    return ratio

if __name__ == "__main__":
    main()