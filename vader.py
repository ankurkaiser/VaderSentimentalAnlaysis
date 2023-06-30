import streamlit as st
import subprocess

# Install necessary dependencies
subprocess.call(["pip", "install", "nltk"])
import nltk
nltk.download('vader_lexicon')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Function to perform sentiment analysis
def analyze_sentiment(sentence):
    # Create a SentimentIntensityAnalyzer object
    sid_obj = SentimentIntensityAnalyzer()

    # Perform sentiment analysis
    sentiment_dict = sid_obj.polarity_scores(sentence)

    # Display sentiment analysis results
    st.write("Overall sentiment dictionary is:", sentiment_dict)
    st.write("Sentence was rated as", sentiment_dict['neg']*100, "% Negative")
    st.write("Sentence was rated as", sentiment_dict['neu']*100, "% Neutral")
    st.write("Sentence was rated as", sentiment_dict['pos']*100, "% Positive")
    st.write("Sentence Overall Rated As", end=" ")

    # Decide sentiment as positive, negative, or neutral
    if sentiment_dict['compound'] >= 0.05:
        st.write("Positive")
    elif sentiment_dict['compound'] <= -0.05:
        st.write("Negative")
    else:
        st.write("Neutral")

# Streamlit app
def main():
    st.title("Sentiment Analysis Tool")
    sentence = st.text_input("Enter a sentence")
    if sentence:
        # Create an empty output box
        output = st.empty()

        # Perform sentiment analysis and update output box
        with output:
            analyze_sentiment(sentence)

        # Reset the input sentence
        st.text_input("Enter a sentence")

if __name__ == "__main__":
    main()
