import streamlit as st
import spacy
from transformers import pipeline

# List of common filler words and phrases to detect in speech
FILLER_WORDS = [
     "um", "uh", "like", "you know", "i mean", "well", "so", "basically",
     "actually", "er", "ah", "hmm", "hm", "you know what i mean", "you know",
     "for sure"
]

# Pre-trained model for sentiment analysis
# This is defined here, so that it doesn't clog standard output
model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

@st.cache_resource
def load_sentiment_model():
    """
    Load and cache the sentiment analysis model.
    
    Returns:
        pipeline: A Hugging Face pipeline for sentiment analysis
    """
    return pipeline("sentiment-analysis", model=model)

def load_transcript(file_path):
    """
    Load and preprocess a transcript file.
    
    Args:
        file_path (str): Path to the transcript file
        
    Returns:
        list: List of non-empty lines from the transcript
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Remove empty lines and strip whitespace
    lines = [line.strip() for line in lines if line.strip()]
    return lines

def compute_sentiment(text):
    """
    Analyze the sentiment of a given text.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: Dictionary containing sentiment label and confidence score
    """
    sentiment_pipeline = load_sentiment_model()
    if not text or text.isspace():
        return {"label": "NEUTRAL", "score": 0.0}
    result = sentiment_pipeline(text)[0]
    return result

def compute_filler_ratio(text):
    """
    Calculate the ratio of filler words in the text and identify which fillers were used.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        tuple: (filler_ratio, list_of_fillers, total_words)
            - filler_ratio (float): Ratio of filler words to total words
            - list_of_fillers (list): List of filler words found in the text
            - total_words (int): Total word count in the text
    """
    total_words = len(text.split())
    fillers = []

    # Use spaCy for tokenization and processing
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text.lower())

    # Count filler words
    filler_count = 0
    for token in doc:
         if token.text in FILLER_WORDS:
              filler_count += 1
              fillers.append(token.text)
    
    ratio = filler_count / total_words if total_words > 0 else 0
    return ratio, fillers, total_words

def analyse_transcript(lines):
    """
    Analyze a complete transcript, processing each line for sentiment and filler words.
    
    Args:
        lines (list): List of transcript lines in format "Speaker: Text"
        
    Returns:
        list: List of dictionaries containing analysis results for each line
            Each dictionary contains:
            - speaker: Speaker identifier
            - text: Original text
            - sentiment_label: Sentiment classification
            - sentiment_score: Confidence score for sentiment
            - filler_ratio: Ratio of filler words
            - filler_words: List of filler words found
            - total_words: Word count
    """
    results = []

    for line in lines:
        # Split line into speaker and text
        speaker = line.split(":")[0]
        text = line.split(":")[1].strip()

        # Perform analysis
        sentiment = compute_sentiment(text)
        sentiment_label = sentiment["label"]
        sentiment_score = sentiment["score"]
        filler_ratio, fillers, total_words = compute_filler_ratio(text)

        # Store results
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
