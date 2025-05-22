import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from analysis import analyse_transcript, load_transcript

st.set_page_config(
    page_title="Bones Transcript Analysis",
    layout="wide"
)

st.title("Bones Transcript Analysis Dashboard")

lines = load_transcript('transcript.txt')
analysis_results = analyse_transcript(lines)

df = pd.DataFrame(analysis_results)

def chat_transcript():
    for index, row in df.iterrows():
        if row['speaker'] == "Speaker A":
            with st.chat_message("A"):
                st.write(row['text'])
                st.write(f"Sentiment: {row['sentiment_label']}")
                st.write(f"Sentiment Score: {row['sentiment_score']}")
                st.write(f"Filler words used: {row['fillers']}")
                st.write(f"Filler Ratio: {row['filler_ratio']}")
        elif row['speaker'] == "Speaker B":
            with st.chat_message("B"):
                st.write(row['text'])
                st.write(f"Sentiment: {row['sentiment_label']}")
                st.write(f"Sentiment Score: {row['sentiment_score']}")
                st.write(f"Filler words used: {row['filler_words']}")
                st.write(f"Filler Ratio: {row['filler_ratio']}")

def dataframe_transcript():
    st.dataframe(df.style.format({
        'sentiment_score': '{:.2f}',
        'filler_ratio' : '{:.2%}'
    }))

chat_display = st.checkbox(label="Chat Display", value=False)

if chat_display:
    chat_transcript()
else:
    dataframe_transcript()

st.subheader("Summary Statistics")

st.markdown("#### Overall Averages")
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Average Sentiment Score", value=f"{df['sentiment_score'].mean():.2f}")
with col2:
    st.metric(label="Average Filler Ratio", value=f"{df['filler_ratio'].mean():.2%}") # Format as percentage

st.markdown("#### Sentiment Distribution")
sentiment_counts = df['sentiment_label'].value_counts()
fig, ax = plt.subplots()
ax.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct='%1.1f%%',
    colors=['#98FF98', '#FF9999', '#99FF99']
)
ax.axis('equal')
st.pyplot(fig)

