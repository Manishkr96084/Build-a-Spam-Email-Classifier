import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df = pd.read_csv("spam_ham_dataset.csv")

if "v1" in df.columns and "v2" in df.columns:
    df = df.rename(columns={"v1": "label", "v2": "text"})

df = df[["text", "label"]]
df["label"] = df["label"].str.lower()

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Type 'stop' to exit.\n")

while True:
    user_input = input("Enter email/message text: ")
    
    if user_input.lower() == "stop":
        print("Program stopped.")
        break
    
    user_input_tfidf = vectorizer.transform([user_input])
    prediction = model.predict(user_input_tfidf)
    
    if prediction[0] == "spam":
        print("Result: SPAM\n")
    else:
        print("Result: HAM\n")
