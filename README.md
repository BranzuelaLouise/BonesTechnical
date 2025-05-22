# BonesTechnical

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/BranzuelaLouise/BonesTechnical.git
cd BonesTechnical
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501` in your web browser.

## Metric Descriptions

### Sentiment Analysis
- **Sentiment Score**: A numerical value between 0 and 1 indicating the confidence of the model in the identified sentiment
  - Values closer to 1 indicate the model is sure about its label
  - Values closer to 0 indicate less confidence in its label
- **Sentiment Label**: Categorical classification of the sentiment (Positive, Negative, or Neutral)

### Filler Word Analysis
- **Filler Words**: Count of filler words used in each speaker's turn
- **Filler Ratio**: Percentage of filler words relative to total words in the turn
- **Recognized Filler Words**: The following words and phrases are counted as filler words in the analysis:
  - Single words: "um", "uh", "like", "well", "so", "basically", "actually", "er", "ah", "hmm", "hm"
  - Phrases: "you know", "i mean", "you know what i mean", "for sure"

### Word Count Statistics
- **Total Words**: Number of words spoken by each participant
- **Average Metrics**: Overall averages for sentiment scores and filler ratios across the entire conversation

## Reflection - In one extra hour
- I would add a better looking and interactive graphs. The graphs from matplotlib
are sufficient, however, I'm not quite satisfied with how they look and present information.
Maybe a few other additions of extra metrics for the transcript analysis.
- Another improvement I would implement
would be a way for someone to upload their own transcript through streamlit, I'd probably have to add a few more
error handling to prevent the loading of the transcript from failing (for example, each line doesn't start with Speaker A:/ Speaker B:).
I could also leverage the text2text-generator pipeline from huggingface to generate transcripts from the press of a button.
- A worry I have is the robustness of the filler word matching. I thought about different ways to create a
good enough approach where it would catch filler words that are sometimes filler words, but not all the time (for example: "like").
I thought about using spacy's part-of-speech tag and syntactic dependencies to identify filler words, but that would have taken a bit
longer to implement. Another idea I had was to just leverage an LLM's ability to identify the filler words, possibly with few-shot
prompting to make the results more consistent.


 
