import importlib
import streamlit as st
import pandas as pd
import re
from collections import Counter
import pickle
import os

try:
    import spacy
except ModuleNotFoundError:
    spacy = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def require_package(package_name, import_name=None):
    try:
        return importlib.import_module(import_name or package_name)
    except ModuleNotFoundError:
        st.error(
            f"Missing Python package: {package_name}.\n"
            "Add it to requirements.txt and deploy again."
        )
        st.stop()

# Set page config
st.set_page_config(
    page_title="🏏 Cricket NLP Analyzer",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for digital look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(45deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        color: white;
    }
    .result-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        text-align: center;
        color: white;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #4facfe;
    }
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #ffa500);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_resource
def load_models():
    # Validate spaCy package and model
    if spacy is None:
        st.error(
            "spaCy is not installed.\n"
            "Add 'spacy' to requirements.txt and deploy again."
        )
        st.stop()

    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        st.error(
            "spaCy model 'en_core_web_sm' is missing.\n"
            "Install it by adding the model to requirements.txt or running: ``python -m spacy download en_core_web_sm``"
        )
        st.stop()

    # Load transformers pipeline lazily
    transformers = require_package('transformers')
    sentiment_pipeline = transformers.pipeline(
        'sentiment-analysis',
        model='distilbert-base-uncased-finetuned-sst-2-english',
        truncation=True,
        max_length=512
    )

    cache_file = 'models_cache.pkl'
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            df, sample_df, tfidf, ml_model, le = pickle.load(f)
    else:
        # Load data
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
        val = pd.read_csv('validation.csv')
        df_raw = pd.concat([train, test, val], ignore_index=True)

        # Parse data
        def parse_row(text):
            text = str(text)
            def get(field):
                m = re.search(field + r' is ([^<]+?)(?= \w+ \w+ is |<end_of_table>)', text)
                return m.group(1).strip() if m else ''
            commentary = ''
            if '<end_of_table>' in text:
                after = text.split('<end_of_table>')[1]
                commentary = after.replace('commentary', '', 1).strip()
            return {
                'play_type': get('play type description'),
                'batting_team': get('batting team'),
                'bowling_team': get('bowling team'),
                'bowler': get('bowler name'),
                'batsman': get('batsman name'),
                'runs': get('total runs on delivery'),
                'dismissal': get('dismissal is'),
                'commentary': commentary
            }

        parsed = df_raw['rows'].apply(parse_row)
        df = pd.DataFrame(list(parsed))

        # Clean text
        def clean_text(text):
            if not isinstance(text, str) or text.strip() == '':
                return ''
            text = re.sub(r'<.*?>', '', text)
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r"[^A-Za-z0-9.,!?'\s]", ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        df['clean_commentary'] = df['commentary'].apply(clean_text)
        df = df[df['clean_commentary'].str.len() > 10].reset_index(drop=True)

        # Train classifier
        clf_df = df[df['play_type'].notna() & (df['play_type'] != '')].copy()
        clf_df = clf_df[clf_df['clean_commentary'].str.len() > 10].copy()

        le = LabelEncoder()
        clf_df['label'] = le.fit_transform(clf_df['play_type'])

        X = clf_df['clean_commentary']
        y = clf_df['label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        ml_model = LogisticRegression(max_iter=300, C=1.0)

        X_train_vec = tfidf.fit_transform(X_train)
        X_test_vec = tfidf.transform(X_test)

        ml_model.fit(X_train_vec, y_train)

        # Compute sentiment for sample
        sample_df = df.head(500).copy()
        sentiments = sample_df['clean_commentary'].apply(lambda x: get_sentiment(x, sentiment_pipeline))
        sample_df['sentiment_label'] = sentiments.apply(lambda x: x[0])
        sample_df['sentiment_score'] = sentiments.apply(lambda x: x[1])

        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump((df, sample_df, tfidf, ml_model, le), f)

    return df, sample_df, nlp, sentiment_pipeline, tfidf, ml_model, le

with st.spinner("Loading NLP models and data... (first run may take a minute, subsequent runs are fast)"):
    df, sample_df, nlp, sentiment_pipeline, tfidf, ml_model, le = load_models()

CRICKET_VENUES = [
    'wankhede', 'eden gardens', 'lords', "lord's", 'oval', 'gabba',
    'mcg', 'scg', 'headingley', 'old trafford', 'chepauk', 'chinnaswamy',
    'feroz shah kotla', 'narendra modi stadium'
]

def extract_entities(text, nlp):
    doc = nlp(text[:300])
    players = []
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            players.append(ent.text.strip())
    venues = [v.title() for v in CRICKET_VENUES if v in text.lower()]
    return list(set(players)), list(set(venues))

def get_sentiment(text, pipeline):
    try:
        result = pipeline(text[:512])[0]
        return result['label'], round(result['score'], 3)
    except:
        return 'NEUTRAL', 0.5

def classify_play(text, tfidf, model, le):
    vec = tfidf.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()
    play = le.inverse_transform([pred])[0]
    return play, round(prob, 3)

# Main app
st.markdown('<h1 class="main-header">🏏 Cricket NLP Analyzer</h1>', unsafe_allow_html=True)
st.markdown("Analyze cricket commentary with AI-powered Named Entity Recognition, Sentiment Analysis, and Play Type Classification!")

# Sidebar
st.sidebar.title("📊 Dataset Overview")
st.sidebar.metric("Total Samples", len(df))
st.sidebar.metric("Unique Batsmen", df['batsman'].nunique())
st.sidebar.metric("Unique Bowlers", df['bowler'].nunique())

# Play type distribution
play_counts = df['play_type'].value_counts()
st.sidebar.subheader("Play Type Distribution")
for play, count in play_counts.items():
    st.sidebar.write(f"{play}: {count}")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Enter Cricket Commentary")
    user_input = st.text_area(
        "Type or paste cricket commentary here:",
        height=150,
        placeholder="e.g., Kohli drives beautifully through the covers for a magnificent four! The crowd erupts at Wankhede."
    )

    if st.button("🚀 Analyze Commentary", key="analyze"):
        if user_input.strip():
            # Clean input
            cleaned = re.sub(r'\s+', ' ', user_input.strip())

            # NER
            players, venues = extract_entities(cleaned, nlp)

            # Sentiment
            sentiment_label, sentiment_score = get_sentiment(cleaned, sentiment_pipeline)

            # Classification
            play_type, play_conf = classify_play(cleaned, tfidf, ml_model, le)

            # Display results
            st.markdown("---")
            st.subheader("📋 Analysis Results")

            # Results cards
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown(f"""
                <div class="result-card">
                    <h3>🔍 Named Entities</h3>
                    <p><strong>Players:</strong> {', '.join(players) if players else 'None detected'}</p>
                    <p><strong>Venues:</strong> {', '.join(venues) if venues else 'None detected'}</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="result-card">
                    <h3>💬 Sentiment</h3>
                    <p><strong>Label:</strong> {'😊 POSITIVE' if sentiment_label == 'POSITIVE' else '😞 NEGATIVE'}</p>
                    <p><strong>Confidence:</strong> {sentiment_score}</p>
                </div>
                """, unsafe_allow_html=True)

            with col_b:
                st.markdown(f"""
                <div class="result-card">
                    <h3>📂 Play Type</h3>
                    <p><strong>Predicted:</strong> {play_type}</p>
                    <p><strong>Confidence:</strong> {play_conf}</p>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.warning("Please enter some commentary text to analyze.")

with col2:
    st.subheader("📈 Quick Stats")
    # Sentiment distribution
    sentiment_counts = sample_df['sentiment_label'].value_counts()
    if not sentiment_counts.empty:
        fig, ax = plt.subplots(figsize=(4, 3))
        colors = ['#2ecc71' if l == 'POSITIVE' else '#e74c3c' for l in sentiment_counts.index]
        ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Sentiment Split (Sample)')
        st.pyplot(fig)

    # Top players
    all_players = []
    for i in range(min(500, len(df))):
        p, _ = extract_entities(df['clean_commentary'].iloc[i], nlp)
        all_players.extend(p)
    player_counts = Counter(all_players).most_common(5)
    if player_counts:
        st.subheader("🏆 Top Mentioned Players")
        for player, count in player_counts:
            st.write(f"{player}: {count}")

# Footer
st.markdown("---")
st.markdown("Built with  using Streamlit, spaCy, Transformers, and scikit-learn")