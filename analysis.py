import streamlit as st
import spacy
from transformers import pipeline

FILLER_WORDS = [
     "um", "uh", "like", "you know", "i mean", "well", "so", "basically",
     "actually", "er", "ah", "hmm", "hm", "you know what i mean", "you know",
     "for sure"
]

model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english" # Prevent clogging the standard output

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model=model)

def load_transcript(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines if line.strip()]
    return lines

def compute_sentiment(text):
    sentiment_pipeline = load_sentiment_model()
    if not text or text.isspace():
        return {"label": "NEUTRAL", "score": 0.0}
    result = sentiment_pipeline(text)[0]
    return result

def compute_filler_ratio(text):
    total_words = len(text.split())
    fillers = []

    filler_count = 0
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text.lower())

    for token in doc:
         if token.text in FILLER_WORDS:
              filler_count += 1
              fillers.append(token.text)
    
    ratio = filler_count / total_words
    return ratio, fillers, total_words

def analyse_transcript(lines):
    results = []

    for line in lines:
        speaker = line.split(":")[0]
        text = line.split(":")[1].strip()

        sentiment = compute_sentiment(text)
        sentiment_label = sentiment["label"]
        sentiment_score = sentiment["score"]
        filler_ratio, fillers, total_words = compute_filler_ratio(text)

        results.append({
            "speaker": speaker,
            "text": text,
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "filler_ratio": filler_ratio,
            "filler_words": fillers,
            "total_words": total_words
        })
    
    return results
