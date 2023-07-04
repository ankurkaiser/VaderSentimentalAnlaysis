import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import word_tokenize, pos_tag
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download necessary resources
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')

# Function to perform sentiment analysis
def analyze_sentiment(text):
    # Create a SentimentIntensityAnalyzer object
    sid_obj = SentimentIntensityAnalyzer()

    # Perform sentiment analysis
    sentiment_dict = sid_obj.polarity_scores(text)

    # Determine mood based on sentiment scores
    if sentiment_dict['compound'] >= 0.05:
        mood = 'happy'
    elif sentiment_dict['compound'] <= -0.05:
        mood = 'sad'
    else:
        mood = 'neutral'

    # Tokenize the text and perform part-of-speech tagging
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)

    # Extract key words based on POS tags
    key_words = [word for word, tag in tagged_tokens if tag.startswith('JJ') or tag.startswith('RB')]

    # Calculate rounded percentages
    pos_percentage = round(sentiment_dict['pos'] * 100, 2)
    neg_percentage = round(sentiment_dict['neg'] * 100, 2)
    neu_percentage = round(sentiment_dict['neu'] * 100, 2)
    overall_score = round((sentiment_dict['pos'] - sentiment_dict['neg'] + 1) / 2, 2)

    # Display sentiment analysis results
    st.write("Sentiment Analysis Results:")
    st.write("Text:", text)
    st.write("Mood:", mood)
    st.write("Positive:", pos_percentage, "%")
    st.write("Negative:", neg_percentage, "%")
    st.write("Neutral:", neu_percentage, "%")
    st.write("Compound:", sentiment_dict['compound'])
    st.write("Overall Sentiment Score:", (overall_score)*100)
    st.write("Key Words:", ', '.join(key_words))

    # Generate and display word cloud
    wordcloud_text = ' '.join(key_words)
    wordcloud = WordCloud(width=800, height=400).generate(wordcloud_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Function to analyze mood based on sentiment analysis
def analyze_mood():
    st.title("Sentimental Analysis App")
    text = st.text_area("Enter some text")
    if st.button("Analyze"):
        analyze_sentiment(text)

# Streamlit app
def main():
    analyze_mood()

if __name__ == "__main__":
    main()
