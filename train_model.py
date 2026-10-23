import pandas as pd

data = pd.read_csv(
    "dataset/spam.csv",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)


print("First 5 rows:")
print(data.head())

data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

print("\nAfter converting labels:")
print(data.head())

print("\nLabel Count:")
print(data["label"].value_counts())

from sklearn.model_selection import train_test_split

X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:", len(X_train))
print("Testing data:", len(X_test))


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF Training Shape:", X_train_tfidf.shape)
print("TF-IDF Testing Shape:", X_test_tfidf.shape)



from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(X_train_tfidf, y_train)

print("\nModel training completed successfully!")



from sklearn.metrics import accuracy_score, classification_report


y_pred = model.predict(X_test_tfidf)


accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")


print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Spam", "Spam"]
))  


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Not Spam", "Spam"],
    yticklabels=["Not Spam", "Spam"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Spam Email Detection - Confusion Matrix")

plt.show()


print("\n--- Spam Email Detector ---")

message = input("Enter your email/message: ")

message_tfidf = vectorizer.transform([message])

prediction = model.predict(message_tfidf)[0]


if prediction == 1:
    print("\nResult: SPAM 🚫")
else:
    print("\nResult: NOT SPAM ✅")