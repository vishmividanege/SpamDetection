📧 Spam Message Detection using Machine Learning

This project is a Spam Detection System that classifies SMS messages as Spam or Not Spam (Ham) using Natural Language Processing (NLP) and various Machine Learning algorithms.

🔹 Features

Dataset Preprocessing

Cleaned SMS data (lowercasing, removing punctuation, numbers, and stopwords).

Lemmatization for better text normalization.

Feature Extraction

Used TF-IDF (Term Frequency – Inverse Document Frequency) to convert text into numerical features.

Machine Learning Models

✅ Naive Bayes Classifier

✅ Logistic Regression

✅ Random Forest Classifier

✅ Support Vector Machine (SVM)

Evaluation

Compared models based on Accuracy, Precision, Recall, F1-score.

Visualized results using Confusion Matrix Heatmap, Correlation Heatmap, and Word Clouds.

Interactive Testing

Added function to test custom messages:

print(predict_spam("Congratulations! You won a free iPhone. Call now!"))  
# 🚫 Spam  

print(predict_spam("Hey, are we meeting tomorrow?"))  
# ✅ Not Spam  


Model Persistence

Saved trained model (spam_model.pkl) and vectorizer (tfidf_vectorizer.pkl) for reuse.

🔹 Visualizations

Confusion Matrix Heatmap

TF-IDF Feature Correlation Heatmap

WordCloud for Spam vs. Ham messages

🔹 Tech Stack

Python

NLTK (Stopwords, Lemmatizer)

scikit-learn (ML models, evaluation)

Seaborn & Matplotlib (visualizations)

WordCloud

📊 Model Performance (Sample)
Model	Accuracy
Naive Bayes	96.2%
Logistic Regression	94.9%
Random Forest	97.5%
Support Vector Machine	97.4%
