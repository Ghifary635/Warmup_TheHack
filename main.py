# ================================
# 📦 Import Libraries
# ================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet, stopwords
import pickle

# ================================
# 🔧 Download NLTK Resources
# ================================
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ================================
# ⚙️ Streamlit Config
# ================================
st.set_page_config(page_title="Ganjar Pranowo Sentiment Analysis", layout="wide")
st.title("Sentiment Analysis of Tweets About Ganjar Pranowo")

# ================================
# 📖 Sidebar
# ================================
st.sidebar.header("About")
st.sidebar.write("""
This app analyzes sentiment of tweets about Indonesian politician Ganjar Pranowo.
The model was trained on 10,000 tweets labeled as Positive or Negative.
""")

# ================================
# 📥 Load Data & Models
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv('Ganjar Pranowo.csv')[['Text', 'label']]
    return df

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

@st.cache_resource
def load_vectorizer():
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        return pickle.load(file)

df = load_data()
model = load_model()
tfidf = load_vectorizer()

# ================================
# 🧹 Text Preprocessing
# ================================
extra_stopwords = [
    'mr', 'will', 'come', 'make', 'know', 'want', 'really', 'must', 'great',
    'time', 'still', 'top', 'two', 'one', 'goodness', 'hopefully', 'god',
    'blessing', 'thank', 'pak', 'u', 'sir'
]
stop_words = set(stopwords.words('english')).union(extra_stopwords)
lemmatizer = WordNetLemmatizer()

def clean_twitter_text(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'RT\s+', '', text)
    text = re.sub(r'[^A-Za-z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

def lemmatize_tokens(tokens):
    tagged = pos_tag(tokens)
    return [lemmatizer.lemmatize(w, get_wordnet_pos(pos)) for w, pos in tagged]

def preprocess_text(text_list):
    processed = []
    tokenizer = RegexpTokenizer(r'\w+')
    for text in text_list:
        cleaned = clean_twitter_text(text.lower())
        tokens = tokenizer.tokenize(cleaned)
        lemmas = lemmatize_tokens(tokens)
        filtered = [w for w in lemmas if w not in stop_words]
        processed.append(' '.join(filtered))
    return processed

# ================================
# 🧩 Tabs Layout
# ================================
tab1, tab2, tab3 = st.tabs(["Project Overview", "Data Exploration", "Sentiment Analysis"])

# Tab 1: Project Overview
with tab1:
    st.header("Project Overview")
    st.write("""
    This project analyzes public sentiment about Ganjar Pranowo through Twitter data. 
    The process includes:
    - Collecting 10,000 tweets
    - Cleaning and preprocessing
    - Sentiment classification using ML
    - Visualization of results
    """)
    
    st.subheader("Methodology")
    st.write("""
    1. Data Collection from Kaggle
    2. Preprocessing (cleaning, tokenizing, lemmatizing, stopword removal)  
    3. Modeling using TF-IDF + Linear SVM  
    4. Evaluation via metrics and confusion matrix  
    """)

# Tab 2: Data Exploration
with tab2:
    st.header("Data Exploration")
    
    st.subheader("Sample Data")
    st.dataframe(df.head())
    
    st.subheader("Class Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='label', ax=ax)
    ax.set_title('Distribution of Sentiment Labels')
    st.pyplot(fig)

# Tab 3: Sentiment Analysis
with tab3:
    st.header("Sentiment Analysis")
    
    user_input = st.text_area("Enter tweet text:", "", key="tweet_input")
    
    if st.button("Analyze Sentiment", key="analyze_btn"):
        if not user_input.strip():
            st.warning("Please enter some text.")
            st.stop()
        
        try:
            processed = preprocess_text([user_input])
            vectorized = tfidf.transform(processed)
            
            prediction = model.predict(vectorized)[0]
            decision_score = model.decision_function(vectorized)[0]
            
            prob_positive = 1 / (1 + np.exp(-decision_score))
            prob_negative = 1 - prob_positive
            
            sentiment = "Positive" if prediction == 1 else "Negative"
            confidence = max(prob_positive, prob_negative) * 100
            
            st.subheader("Result")
            st.metric("Sentiment", sentiment)
            st.metric("Confidence", f"{confidence:.1f}%")
            
            chart_df = pd.DataFrame({
                "Sentiment": ["Negative", "Positive"],
                "Probability": [prob_negative, prob_positive]
            })
            st.bar_chart(chart_df.set_index("Sentiment"), use_container_width=True)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
