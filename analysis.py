import pandas as pd
from transformers import pipeline

def main():
    lines = load_transcript("transcript.txt")

    for line in lines:
        sentiment = compute_sentiment(line)
        print(sentiment)

def load_transcript(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines if line.strip()]
    return lines

def compute_sentiment(text):
        sentiment_pipeline = pipeline("sentiment-analysis")
        if not text or text.isspace():
            return {"label": "NEUTRAL", "score": 0.0}
        result = sentiment_pipeline(text)[0]
        return result

if __name__ == "__main__":
    main()