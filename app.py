import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model and vectorizer
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# Streamlit UI
st.set_page_config(page_title="📧 Spam Detector", page_icon="📨", layout="centered")
st.title("📧 Creative Spam Detection App")
st.write("Type a message below and see how spammy it is! 🕵️‍♂️")

# Input area
user_input = st.text_area("Type your message here:")

# Prediction
if st.button("🔍 Check Spam Level"):
    if user_input.strip() != "":
        processed = preprocess_text(user_input)
        vect = vectorizer.transform([processed])
        prediction = model.predict(vect)[0]
        probability = model.predict_proba(vect)[0][1] * 100  # spam probability

        # Display spam meter
        st.subheader("📊 Spam Meter")
        st.progress(int(probability))
        st.write(f"Spam Probability: {probability:.2f}%")

        # Fun messages based on probability
        if probability > 75:
            st.markdown("<h3 style='color:red'>🚨 Highly Spammy Message!</h3>", unsafe_allow_html=True)
            st.image("https://media.giphy.com/media/26xBwdIuRJiAi7nU0/giphy.gif")
        elif probability > 40:
            st.markdown("<h3 style='color:orange'>⚠️ Possibly Spam</h3>", unsafe_allow_html=True)
            st.image("https://media.giphy.com/media/l0HlBO7eyXzSZkJri/giphy.gif")
        else:
            st.markdown("<h3 style='color:green'>✅ Safe Message</h3>", unsafe_allow_html=True)
            st.image("https://media.giphy.com/media/111ebonMs90YLu/giphy.gif")
    else:
        st.error("Please enter a message!")






