import streamlit as st
import pandas as pd
import plotly.express as px
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from transformers import pipeline

# Set your API key here
YOUTUBE_API_KEY = "AIzaSyDjGOZQhzqQvZhfMBA9P2nwgr66GBQ2bQ0"  # Replace with your actual API key

# Initialize models
sentiment_analyzer = None
emotion_analyzer = None
sarcasm_detector = None

# Load models once to optimize memory usage
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0)
    emotion_analyzer = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
    sarcasm_detector = pipeline("text-classification", model="distilbert-base-uncased", device=0)
except Exception as e:
    st.error(f"Error loading models: {e}")

# Function to analyze sentiment
def analyze_sentiment(text: str):
    if sentiment_analyzer:
        sentiment = sentiment_analyzer(text)[0]
        return sentiment['label'], sentiment['score']
    return "Error", 0

# Function to analyze emotions
def analyze_emotion(text: str):
    if emotion_analyzer:
        candidate_labels = ["joy", "anger", "fear", "sadness", "surprise"]
        result = emotion_analyzer(text, candidate_labels=candidate_labels)
        return result['labels'][0], result['scores'][0]
    return "Error", 0

# Function to detect sarcasm
def detect_sarcasm(text: str):
    if sarcasm_detector:
        result = sarcasm_detector(text)
        return result[0]['label']
    return "Error"

# Function to split long comments into smaller chunks
def split_text(text, max_length=512):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(' '.join(current_chunk)) > max_length:
            chunks.append(' '.join(current_chunk[:-1]))  # Exclude the last word
            current_chunk = [word]  # Start new chunk with the current word

    if current_chunk:
        chunks.append(' '.join(current_chunk))  # Add the remaining words as the last chunk

    return chunks

# Function to analyze the dynamic sentiment of the text
def analyze_dynamic_sentiment(text: str):
    chunks = split_text(text)

    # Process each chunk and combine the results
    sentiment_label, sentiment_score = None, 0
    emotion_label, emotion_score = None, 0
    sarcasm_label = None

    for chunk in chunks:
        sentiment_chunk = analyze_sentiment(chunk)
        sentiment_label = sentiment_chunk[0]
        sentiment_score += sentiment_chunk[1]

        emotion_chunk = analyze_emotion(chunk)
        emotion_label = emotion_chunk[0]
        emotion_score += emotion_chunk[1]

        sarcasm_chunk = detect_sarcasm(chunk)
        sarcasm_label = sarcasm_chunk

    # Average the sentiment and emotion scores (if there were multiple chunks)
    sentiment_score /= len(chunks)
    emotion_score /= len(chunks)

    response = {
        "sentiment": sentiment_label,
        "sentiment_score": sentiment_score,
        "emotion": emotion_label,
        "emotion_score": emotion_score,
        "sarcasm": sarcasm_label
    }

    return response

# Function to extract comments from a YouTube video using the YouTube Data API
def get_youtube_comments(video_url: str):
    video_id = video_url.split('v=')[1].split('&')[0]

    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    comments = []
    try:
        comment_request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=50  # Limit to avoid large data processing
        )
        comment_response = comment_request.execute()

        for item in comment_response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

    except HttpError as e:
        st.error(f"An error occurred while fetching comments: {e}")

    return comments

# Streamlit UI
st.title("YouTube Comment Sentiment, Emotion, and Sarcasm Analyzer")

# User input for YouTube video link
video_url = st.text_input("Please enter the YouTube video link:")

if video_url:
    comments = get_youtube_comments(video_url)

    if comments:
        st.write("Comments extracted from the video:")

        # Display each comment and its analysis
        results = []
        emotion_counts = {"joy": 0, "anger": 0, "fear": 0, "sadness": 0, "surprise": 0}

        for comment in comments:
            sentiment = analyze_dynamic_sentiment(comment)
            results.append({
                "Comment": comment,
                "Sentiment Analysis": sentiment['sentiment'],
                "Sentiment Score": sentiment['sentiment_score'],
                "Emotion": sentiment['emotion'],
                "Emotion Score": sentiment['emotion_score'],
                "Sarcasm": sentiment['sarcasm']
            })

            # Count emotions for the pie chart
            emotion_counts[sentiment['emotion']] += 1

        # Convert to DataFrame for easy viewing
        df_results = pd.DataFrame(results)

        # Display the DataFrame in Streamlit
        st.write("Sentiment and Emotion Analysis of YouTube Comments:")
        st.dataframe(df_results)

        # Plot pie chart for emotion distribution
        emotion_color_map = {
            "joy": "yellow",
            "anger": "red",
            "fear": "purple",
            "sadness": "blue",
            "surprise": "orange"
        }

        fig = px.pie(
            names=list(emotion_counts.keys()),
            values=list(emotion_counts.values()),
            title=f"Distribution of Emotions in {len(comments)} YouTube Comments",
            color=list(emotion_counts.keys()),
            color_discrete_map=emotion_color_map
        )

        fig.update_traces(
            textinfo='percent+label',
            hovertemplate="%{label}: %{percent:.1%} (%{value} comments)"
        )

        # Update layout to expand legend
        fig.update_layout(
            legend=dict(
                title="Emotions",
                orientation="h",  # Horizontal layout for the legend
                yanchor="bottom",
                y=-0.2,  # Adjust position of the legend
                xanchor="center",
                x=0.5
            )
        )

        st.plotly_chart(fig)

        # Display sentiment summary table
        st.write("### Sentiment Summary Table")
        sentiment_summary = df_results['Sentiment Analysis'].value_counts().to_dict()
        sentiment_summary_df = pd.DataFrame(list(sentiment_summary.items()), columns=["Sentiment", "Count"])
        st.dataframe(sentiment_summary_df)

    else:
        st.write("No comments found for the video.")
