import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

# NLTK setup - explicit downloads at the beginning
import nltk
try:
    # Explicitly download required NLTK resources
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')  # Open Multilingual WordNet data
    
    # Import NLTK modules after downloading resources
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except Exception as e:
    st.error(f"Error downloading NLTK resources: {e}")
    st.info("If you're still experiencing issues, run these commands in a Python terminal:\n```\nimport nltk\nnltk.download('punkt')\nnltk.download('stopwords')\nnltk.download('wordnet')\nnltk.download('omw-1.4')\n```")

# Set page config
st.set_page_config(
    page_title="Disaster Tweets Classifier",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to clean text
def clean_text(text):
    """Clean and preprocess text data"""
    # Convert to string
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Convert to lowercase
    text = text.lower()
    
    try:
        # Tokenize text
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        # Rejoin tokens
        text = ' '.join(tokens)
    except Exception as e:
        st.warning(f"Error in text processing:  Using simple tokenization instead.")
        # Fallback to simple tokenization if NLTK fails
        tokens = text.split()
        text = ' '.join(tokens)
    
    return text

# Function to predict class
def predict_disaster(tweet, model, vectorizer):
    """Predict if a tweet is about a disaster"""
    # Clean the tweet
    cleaned_tweet = clean_text(tweet)
    
    # Vectorize the tweet
    tweet_vector = vectorizer.transform([cleaned_tweet])
    
    # Make prediction
    prediction = model.predict(tweet_vector)[0]
    
    # Get probability if available
    probability = None
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(tweet_vector)[0][prediction]
    
    # Return prediction, probability, and cleaned tweet
    return prediction, probability, cleaned_tweet

# Function to load models
@st.cache_resource
def load_models(model_path='disaster_tweets_model.pkl', vectorizer_path='tfidf_vectorizer.pkl'):
    """Load the model and vectorizer from pickle files"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Title and description
st.title("🚨 Disaster Tweets Classifier")
st.markdown("""
This app uses a machine learning model to predict whether a tweet is about a real disaster or not.
Enter a tweet below or try one of the example tweets!
""")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
This application uses a machine learning model trained on thousands of tweets to identify which ones are about real disasters.

### How it works:
1. Enter a tweet or select an example
2. The model preprocesses the text
3. The classifier predicts whether it's about a disaster
4. Results are displayed with confidence score

### Model details:
- Uses TF-IDF vectorization
- Trained on 7,613 labeled tweets
- Includes comprehensive text preprocessing
""")


example_tweets = [
    "BREAKING: Massive earthquake hits coast, thousands feared dead.",
    "Just had an amazing dinner at the new restaurant downtown! #foodie",
    "URGENT: Forest fire spreading rapidly, evacuations underway.",
    "Can't wait for the new Avengers movie to drop! So excited!",
    "WARNING: Hurricane heading towards Florida, expected to make landfall tomorrow."
]



# Load models
model, vectorizer = load_models()

if model is None or vectorizer is None:
    st.error("Failed to load models. Please make sure the model files exist and are in the correct format.")
    st.info("Model files needed: 'disaster_tweets_model.pkl' and 'tfidf_vectorizer.pkl'")
    st.stop()

# Create tabs
tab1, tab2 = st.tabs(["Single Tweet Prediction", "Batch Prediction"])

# Tab 1: Single Tweet Prediction
with tab1:
    # Input for user to enter a tweet
   
    
    tweet_input = st.text_area("Enter a tweet:", height=100, 
                                  placeholder="Enter tweet text here...")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        predict_button = st.button("Predict", type="primary")
    
    with col2:
        clear_button = st.button("Clear")
    
    if clear_button:
        tweet_input = ""
        st.experimental_rerun()
    
    # Make prediction when button is clicked
    if predict_button and tweet_input:
        with st.spinner("Analyzing tweet..."):
            # Make prediction
            prediction, probability, cleaned_tweet = predict_disaster(tweet_input, model, vectorizer)
            
            # Display results
            st.subheader("Results")
            
            # Create columns for display
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if prediction == 1:
                    st.markdown("### 🚨 **DISASTER**")
                    st.markdown(f"This tweet is about a real disaster.")
                else:
                    st.markdown("### ✅ **NOT A DISASTER**")
                    st.markdown(f"This tweet is not about a real disaster.")
                
                if probability is not None:
                    st.markdown(f"**Confidence:** {probability:.2f}")
            
            with col2:
                # Create a gauge chart for confidence
                if probability is not None:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    
                    # Create background circle
                    ax.add_patch(plt.Circle((0.5, 0), 0.4, color='lightgray', zorder=0))
                    
                    # Create colored arc based on confidence
                    confidence_color = 'red' if prediction == 1 else 'green'
                    theta = np.linspace(np.pi, 0, 100)
                    x = 0.5 + 0.4 * np.cos(theta)
                    y = 0 + 0.4 * np.sin(theta)
                    ax.plot(x, y, color='lightgray', linewidth=10, solid_capstyle='round')
                    
                    end_angle = np.pi - np.pi * probability
                    theta = np.linspace(np.pi, end_angle, 100)
                    x = 0.5 + 0.4 * np.cos(theta)
                    y = 0 + 0.4 * np.sin(theta)
                    ax.plot(x, y, color=confidence_color, linewidth=10, solid_capstyle='round')
                    
                    # Add confidence text
                    ax.text(0.5, 0, f"{probability:.2f}", ha='center', va='center', fontsize=20)
                    ax.text(0.5, -0.15, "Confidence", ha='center', fontsize=12)
                    
                    # Set limits and remove axes
                    ax.set_xlim(0, 1)
                    ax.set_ylim(-0.5, 0.5)
                    ax.axis('off')
                    
                    st.pyplot(fig)
            
            # Show text processing details
            with st.expander("View text processing details"):
                st.markdown("**Original Text:**")
                st.write(tweet_input)
                
                st.markdown("**Cleaned Text:**")
                st.write(cleaned_tweet)
                
                st.markdown("**Processing Steps:**")
                st.markdown("""
                1. Removed URLs, mentions, and hashtags
                2. Removed punctuation and numbers
                3. Converted to lowercase
                4. Removed stopwords (common words like 'the', 'a', etc.)
                5. Applied lemmatization (reducing words to their base form)
                """)

# Tab 2: Batch Prediction
with tab2:
    st.header("Batch Prediction")
    st.markdown("""
    Upload a CSV file to predict multiple tweets at once. 
    The file should have a column named 'text' containing the tweets.
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'text' not in df.columns:
                st.error("CSV file must contain a column named 'text'")
                st.info("Available columns: " + ", ".join(df.columns))
            else:
                if st.button("Run Batch Prediction", type="primary"):
                    with st.spinner("Processing tweets..."):
                        # Make predictions
                        results = []
                        progress_bar = st.progress(0)
                        total_rows = len(df)
                        
                        for idx, row in df.iterrows():
                            tweet = row['text']
                            prediction, probability, cleaned_tweet = predict_disaster(tweet, model, vectorizer)
                            results.append({
                                'original_text': tweet,
                                'cleaned_text': cleaned_tweet,
                                'prediction': "Disaster" if prediction == 1 else "Not Disaster",
                                'confidence': probability if probability is not None else None
                            })
                            # Update progress bar
                            progress_bar.progress((idx + 1) / total_rows)
                        
                        results_df = pd.DataFrame(results)
                        
                        # Display results
                        st.subheader("Prediction Results")
                        st.dataframe(results_df)
                        
                        # Plot class distribution
                        st.subheader("Prediction Distribution")
                        fig, ax = plt.subplots(figsize=(8, 4))
                        prediction_counts = results_df['prediction'].value_counts()
                        colors = ['green', 'red'] if len(prediction_counts) > 1 else ['red'] if 'Disaster' in prediction_counts.index else ['green']
                        sns.barplot(x=prediction_counts.index, y=prediction_counts.values, palette=colors)
                        ax.set_xlabel("Prediction")
                        ax.set_ylabel("Count")
                        st.pyplot(fig)
                        
                        # Option to download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="disaster_predictions.csv",
                            mime="text/csv"
                        )
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("Make sure your CSV file is properly formatted and contains tweet text.")



