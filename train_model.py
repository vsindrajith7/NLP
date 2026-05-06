import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load and preprocess data (same as in app.py)
def load_and_preprocess_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    val = pd.read_csv('validation.csv')
    df_raw = pd.concat([train, test, val], ignore_index=True)

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
    # Sample 10% for faster training
    df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)
    return df

# Train classifier
def train_classifier(df):
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

    return tfidf, ml_model, le

# Load data and train
df = load_and_preprocess_data()
tfidf, ml_model, le = train_classifier(df)

# Save the model
with open('classifier.pkl', 'wb') as f:
    pickle.dump((tfidf, ml_model, le), f)
print("Model saved as classifier.pkl")