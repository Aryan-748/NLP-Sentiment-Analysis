import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Download NLTK resources
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# Load dataset
df = pd.read_csv("TestReviews.csv")

print("Dataset Preview")
print(df.head())

# Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["clean_review"] = df["review"].apply(preprocess)

print("\nCleaned Reviews")
print(df[["review", "clean_review"]].head())

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(df["clean_review"])

y = df["class"]

print("\nTF-IDF Shape:", X.shape)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = MultinomialNB()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Classification Report
print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative","Positive"],
    yticklabels=["Negative","Positive"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")



plt.show()

# TF-IDF Visualization
feature_names = tfidf.get_feature_names_out()

importance = X.toarray().sum(axis=0)

top = pd.DataFrame({
    "Word": feature_names,
    "Score": importance
})

top = top.sort_values("Score", ascending=False).head(15)

plt.figure(figsize=(10,5))

sns.barplot(
    data=top,
    x="Score",
    y="Word"
)

plt.title("Top TF-IDF Features")

plt.tight_layout()



plt.show()

# Sample Predictions
sample_reviews = [
    "The food was amazing and the staff was friendly.",
    "Worst experience ever.",
    "I really enjoyed the service.",
    "The product quality is terrible."
]

sample_clean = [preprocess(i) for i in sample_reviews]

sample_vector = tfidf.transform(sample_clean)

predictions = model.predict(sample_vector)

print("\nSample Predictions")

for review, pred in zip(sample_reviews, predictions):

    sentiment = "Positive" if pred == 1 else "Negative"

    print(f"\nReview: {review}")
    print("Prediction:", sentiment)

# Save predictions
prediction_df = pd.DataFrame({
    "Review": sample_reviews,
    "Prediction": [
        "Positive" if i == 1 else "Negative"
        for i in predictions
    ]
})

prediction_df.to_csv(
    "Sample_Predictions.csv",
    index=False
)

print("\nProject Completed Successfully!")