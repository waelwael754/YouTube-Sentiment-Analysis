

import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="YouTube Sentiment Analysis", page_icon="🎬", layout="centered")

st.markdown("""
<style>
.main{background:#0f0f0f;}
h1{text-align:center;color:white;}
h3{text-align:center;color:#ff0000;}
.stTextArea textarea{
background:#1f1f1f;
color:white;
border:2px solid #ff0000;
border-radius:12px;
}
.stButton>button{
width:100%;
height:55px;
background:#ff0000;
color:white;
border:none;
border-radius:12px;
font-size:20px;
font-weight:bold;
}
.stButton>button:hover{
background:#cc0000;
color:white;
}
.result-card{
padding:18px;
border-radius:12px;
font-size:22px;
font-weight:bold;
text-align:center;
margin-top:20px;
}
.positive{background:#0f5132;color:#fff;}
.negative{background:#842029;color:#fff;}
.neutral{background:#0c4a6e;color:#fff;}
</style>
""", unsafe_allow_html=True)

for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet"):
    nltk.download(pkg, quiet=True)
    
model = joblib.load("svm_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
encoder = joblib.load("label_encoder.pkl")

lemmatizer = WordNetLemmatizer()
stopwords_set = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stopwords_set]
    return " ".join(tokens)

st.markdown("# 🎬 YouTube Sentiment Analysis")
st.markdown("### AI Powered Comment Analyzer")
st.write("Analyze YouTube comments using the trained Linear SVM model.")

comment = st.text_area("Enter YouTube Comment", height=180, placeholder="Type your comment here...")

c1,c2 = st.columns(2)
analyze = c1.button("🔍 Analyze")
clear = c2.button("🧹 Clear")

if clear:
    st.rerun()

if analyze:
    if not comment.strip():
        st.warning("Please enter a comment.")
    else:
        with st.spinner("Analyzing..."):
            cleaned = clean_text(comment)
            vec = tfidf.transform([cleaned])
            pred = model.predict(vec)
            sentiment = encoder.inverse_transform(pred)[0]

        if sentiment == "Positive":
            st.markdown('<div class="result-card positive">😊 Positive Comment</div>', unsafe_allow_html=True)
        elif sentiment == "Negative":
            st.markdown('<div class="result-card negative">😞 Negative Comment</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-card neutral">😐 Neutral Comment</div>', unsafe_allow_html=True)