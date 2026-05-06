import streamlit as st
import pandas as pd
import re
import spacy
from collections import Counter
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Cricket NLP Analyzer",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for digitalized look
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    .stSidebar {
        background: rgba(0, 0, 0, 0.8);
        border-right: 2px solid #00ff88;
    }
    .stButton>button {
        background: linear-gradient(45deg, #00ff88, #00b4d8);
        color: black;
        border: none;
        border-radius: 5px;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        border: 1px solid #00ff88;
    }
    .stDataFrame {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #00ff88;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #00ff88;
    }
</style>
""", unsafe_allow_html=True)

# Constants
CRICKET_VENUES = [
    'wankhede', 'eden gardens', 'lords', "lord's", 'oval', 'gabba',
    'mcg', 'scg', 'headingley', 'old trafford', 'chepauk', 'chinnaswamy',
    'feroz shah kotla', 'narendra modi stadium'
]

@st.cache_resource
def load_models():
    """Load and cache NLP models"""
    nlp = spacy.load('en_core_web_sm')
    sentiment_pipeline = pipeline(
        'sentiment-analysis',
        model='distilbert-base-uncased-finetuned-sst-2-english',
        truncation=True,
        max_length=512
    )
    return nlp, sentiment_pipeline

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the cricket data"""
    # Load data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    val = pd.read_csv('validation.csv')
    df_raw = pd.concat([train, test, val], ignore_index=True)

    # Parse rows
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

    return df

@st.cache_resource
def load_classifier():
    """Load the pre-trained classifier"""
    with open('classifier.pkl', 'rb') as f:
        tfidf, ml_model, le = pickle.load(f)
    return tfidf, ml_model, le

def extract_entities(row, nlp):
    """Extract entities from a row"""
    players = [p for p in [row['batsman'], row['bowler']] if p]
    teams = [t for t in [row['batting_team'], row['bowling_team']] if t]

    doc = nlp(row['clean_commentary'][:300])
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            players.append(ent.text.strip())

    venues = [v.title() for v in CRICKET_VENUES if v in row['clean_commentary'].lower()]

    return {
        'ner_players': list(set(players)),
        'ner_teams': list(set(teams)),
        'ner_venues': list(set(venues))
    }

def get_sentiment(text, sentiment_pipeline):
    """Get sentiment of text"""
    try:
        result = sentiment_pipeline(text[:512])[0]
        return result['label'], round(result['score'], 3)
    except:
        return 'NEUTRAL', 0.5

def analyze_commentary(text, nlp, sentiment_pipeline, tfidf, ml_model, le):
    """Analyze a single commentary"""
    cleaned = re.sub(r'\s+', ' ', text.strip())

    # NER
    doc = nlp(cleaned)
    players = list(set([e.text for e in doc.ents if e.label_ == 'PERSON']))
    venues = [v.title() for v in CRICKET_VENUES if v in cleaned.lower()]

    # Sentiment
    label, score = get_sentiment(cleaned, sentiment_pipeline)

    # Classification
    vec = tfidf.transform([cleaned])
    pred = ml_model.predict(vec)[0]
    prob = ml_model.predict_proba(vec).max()
    play = le.inverse_transform([pred])[0]

    return {
        'cleaned_text': cleaned,
        'players': players,
        'venues': venues,
        'sentiment': label,
        'confidence': score,
        'play_type': play,
        'play_confidence': prob
    }

def main():
    # Load models and data
    nlp, sentiment_pipeline = load_models()
    df = load_and_preprocess_data()
    tfidf, ml_model, le = load_classifier()

    # Sidebar
    st.sidebar.title("🏏 Cricket NLP Analyzer")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigation", ["Data Overview", "Analyze Commentary", "Results"])

    if page == "Data Overview":
        st.title("📊 Data Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Unique Batsmen", df["batsman"].nunique())
        with col3:
            st.metric("Unique Bowlers", df["bowler"].nunique())
        with col4:
            st.metric("Play Types", len(df["play_type"].unique()))

        st.markdown("### Play Type Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        play_counts = df['play_type'].value_counts()
        sns.barplot(x=play_counts.index, y=play_counts.values, ax=ax, palette="viridis")
        ax.set_title('Play Type Distribution')
        ax.set_xlabel('Play Type')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

        st.markdown("### Commentary Length Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        df['text_len'] = df['clean_commentary'].apply(len)
        sns.histplot(df['text_len'], bins=40, ax=ax, color='orange')
        ax.set_title('Commentary Text Length')
        ax.set_xlabel('Characters')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

    elif page == "Analyze Commentary":
        st.title("🔍 Analyze Commentary")

        user_input = st.text_area(
            "Enter cricket commentary:",
            height=100,
            placeholder="e.g., Kohli drives beautifully through the covers for a magnificent four! The crowd erupts at Wankhede."
        )

        if st.button("Analyze", type="primary"):
            if user_input.strip():
                with st.spinner("Analyzing..."):
                    result = analyze_commentary(user_input, nlp, sentiment_pipeline, tfidf, ml_model, le)

                st.success("Analysis Complete!")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 📝 Input Text")
                    st.write(result['cleaned_text'])

                    st.markdown("### 🔍 Named Entities")
                    st.write(f"**Players:** {result['players'] if result['players'] else 'None detected'}")
                    st.write(f"**Venues:** {result['venues'] if result['venues'] else 'None detected'}")

                with col2:
                    st.markdown("### 💬 Sentiment")
                    emoji = '😊' if result['sentiment'] == 'POSITIVE' else '😞' if result['sentiment'] == 'NEGATIVE' else '😐'
                    st.write(f"{emoji} **{result['sentiment']}** (confidence: {result['confidence']})")

                    st.markdown("### 📂 Play Type")
                    st.write(f"**{result['play_type']}** (confidence: {result['play_confidence']:.2%})")

            else:
                st.error("Please enter some commentary text.")

    elif page == "Results":
        st.title("📋 Analysis Results")

        # Load results if exists, else generate sample
        try:
            results_df = pd.read_csv('cricket_nlp_results.csv')
        except:
            # Generate sample results
            sample_df = df.head(500).copy()
            ner_results = sample_df.apply(lambda row: extract_entities(row, nlp), axis=1)
            sample_df['ner_players'] = ner_results.apply(lambda x: x['ner_players'])
            sample_df['ner_teams'] = ner_results.apply(lambda x: x['ner_teams'])
            sentiments = sample_df['clean_commentary'].apply(lambda x: get_sentiment(x, sentiment_pipeline))
            sample_df['sentiment_label'] = sentiments.apply(lambda x: x[0])
            sample_df['sentiment_score'] = sentiments.apply(lambda x: x[1])
            results_df = sample_df[['clean_commentary','ner_players','ner_teams','sentiment_label','sentiment_score','play_type']].copy()
            results_df.columns = ['Commentary','Players','Teams','Sentiment','Confidence','Play Type']

        st.dataframe(results_df, use_container_width=True)

        # Download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="cricket_nlp_results.csv",
            mime="text/csv"
        )

    # Footer
    st.markdown("---")
    st.markdown("**Cricket NLP Analyzer** - Built with Streamlit | Team: Adhi · Khaise · Meera · Indrajith")

if __name__ == "__main__":
    main()