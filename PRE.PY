import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import nltk
from nltk.corpus import stopwords

# Load data (replace with your data loading method)
data = pd.read_csv("dataset1.csv")
text = data["text"]
labels = data["reviews"]  # Assuming the label column contains sentiment (positive/negative)

nltk.download('stopwords')

# Define negation words
negations = set(["not", "no", "never", "n't", "couldn't", "didn't", "wouldn't", "won't", "can't", "aren't"])

# Preprocess text data with negation handling
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove punctuation and special characters
    words = text.split()
    processed_words = []
    is_negated = False
    for word in words:
        if word in negations:
            is_negated = True
            continue
        if is_negated:
            word = "NOT_" + word
            is_negated = False
        processed_words.append(word)
    processed_text = " ".join(processed_words)
    return processed_text

# Train-Test Split for Training and Evaluation
X_train, X_test, y_train, y_test = train_test_split(text, labels, test_size=0.2, random_state=42)
preprocessed_train_text = X_train.apply(preprocess_text)
preprocessed_test_text = X_test.apply(preprocess_text)

# Train vectorizer
vectorizer = TfidfVectorizer(max_features=2000)
vectorizer.fit(preprocessed_train_text)

# Train model
model = LogisticRegression(solver="lbfgs", multi_class="multinomial")
model.fit(vectorizer.transform(preprocessed_train_text), y_train)

# Function to classify sentiment with negation handling
def classify_sentiment(text_to_classify, model, vectorizer):
    preprocessed_text = preprocess_text(text_to_classify)
    features = vectorizer.transform([preprocessed_text])
    prediction = model.predict(features)[0]
    return prediction

# Get user input for sentiment classification
while True:
    new_text = input("Enter a review to classify (or 'q' to quit): ")
    if new_text.lower() == 'q':
        break

    sentiment = classify_sentiment(new_text, model, vectorizer)
    print("Predicted sentiment:", sentiment)
