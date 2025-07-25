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
    
    # Tambahkan bagian baru untuk menampilkan kata-kata yang sering muncul
    st.subheader("Most Influential Words by Sentiment")
    
    # Dapatkan fitur names dari TF-IDF
    feature_names = tfidf.get_feature_names_out()
    
    # Dapatkan coefficients dari model SVM
    if hasattr(model, 'coef_'):
        coefficients = model.coef_[0]
    else:  # Jika menggunakan pipeline, akses model-nya terlebih dahulu
        coefficients = model.named_steps['classifier'].coef_[0]
    
    # Buat DataFrame untuk weights
    weights_df = pd.DataFrame({
        'Word': feature_names,
        'Weight': coefficients
    })
    
    # Urutkan berdasarkan weight
    top_positive = weights_df.sort_values('Weight', ascending=False).head(10)
    top_negative = weights_df.sort_values('Weight', ascending=True).head(10)
    
    # Buat visualisasi
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top 10 Positive Words (Most Influential)**")
        fig_pos, ax_pos = plt.subplots(figsize=(8, 5))
        sns.barplot(data=top_positive, y='Word', x='Weight', ax=ax_pos, palette='Blues_d')
        ax_pos.set_xlabel('Weight (Positive Influence)')
        ax_pos.set_ylabel('')
        st.pyplot(fig_pos)
        st.dataframe(top_positive)
    
    with col2:
        st.markdown("**Top 10 Negative Words (Most Influential)**")
        fig_neg, ax_neg = plt.subplots(figsize=(8, 5))
        sns.barplot(data=top_negative, y='Word', x='Weight', ax=ax_neg, palette='Reds_d')
        ax_neg.set_xlabel('Weight (Negative Influence)')
        ax_neg.set_ylabel('')
        st.pyplot(fig_neg)
        st.dataframe(top_negative)

# Tab 3: Sentiment Analysis
with tab3:
    st.header("Sentiment Analysis")

    input_mode = st.radio("Select input type:", ["Manual Text", "Upload CSV"], key="input_mode")

    if input_mode == "Manual Text":
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

    else:  # Upload CSV
        uploaded_file = st.file_uploader("Upload a CSV file with a column named 'text'", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'text' not in df.columns:
                    st.error("The uploaded CSV must contain a column named 'text'.")
                    st.stop()

                processed_texts = preprocess_text(df['text'].tolist())
                vectorized = tfidf.transform(processed_texts)

                predictions = model.predict(vectorized)
                scores = model.decision_function(vectorized)

                prob_pos = 1 / (1 + np.exp(-scores))
                prob_neg = 1 - prob_pos

                df['Sentiment'] = ["Positive" if p == 1 else "Negative" for p in predictions]
                df['Confidence'] = np.maximum(prob_pos, prob_neg) * 100

                st.success("Prediction completed!")
                st.dataframe(df[['text', 'Sentiment', 'Confidence']])

                csv_output = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_output,
                    file_name="sentiment_results.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"An error occurred while processing the file: {str(e)}")
