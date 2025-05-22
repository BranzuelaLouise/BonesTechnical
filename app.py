import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from analysis import analyse_transcript, load_transcript

# Configure the Streamlit page
st.set_page_config(
    page_title="Bones Transcript Analysis",
    layout="wide"
)

st.title("Bones Transcript Analysis Dashboard")

# Load and analyze the transcript
lines = load_transcript('transcript.txt')
analysis_results = analyse_transcript(lines)

# Convert results to DataFrame for easier manipulation
df = pd.DataFrame(analysis_results)

def chat_transcript():
    """
    Display the transcript in a chat-like interface with analysis results.
    Each message shows the text, sentiment, and filler word information.
    """
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
    """
    Display the transcript analysis in a tabular format.
    Formats sentiment scores and filler ratios for better readability.
    """
    st.dataframe(df.style.format({
        'sentiment_score': '{:.2f}',
        'filler_ratio' : '{:.2%}'
    }))

# Toggle between chat and table views
chat_display = st.checkbox(label="Chat Display", value=False)

if chat_display:
    chat_transcript()
else:
    dataframe_transcript()

# Display summary statistics
st.subheader("Summary Statistics")

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Overall Averages", "Sentiment Distribution", "Total Words per Speaker"])

with tab1:
    """Display overall average metrics"""
    st.markdown("#### Overall Averages")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Average Sentiment Score", value=f"{df['sentiment_score'].mean():.2f}")
    with col2:
        st.metric(label="Average Filler Ratio", value=f"{df['filler_ratio'].mean():.2%}")

with tab2:
    """Display sentiment distribution pie chart"""
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

with tab3:
    """Display word count bar chart"""
    st.markdown("#### Total Words per Speaker")

    speaker_word_counts = df.groupby('speaker')['total_words'].sum()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(speaker_word_counts.index, speaker_word_counts.values,
                  color=['#FF9999', '#66B3FF'], alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height - 5,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Total Words', fontsize=12)
    ax.set_title('Total Word Count per Speaker', fontsize=14, fontweight='bold')

    st.pyplot(fig)