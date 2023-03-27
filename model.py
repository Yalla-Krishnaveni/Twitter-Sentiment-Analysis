import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("Tweets.csv")


path = "Tweets.csv"


def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna()
    df = df.drop(['textID'], axis=1)
    LE = preprocessing.LabelEncoder()
    df.sentiment = LE.fit_transform(df.sentiment)
    X = df.selected_text.values
    y = df.sentiment.values
    return X, y


url = "Tweets.csv"
X, y = load_data(url)


# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)


vect = CountVectorizer()  # gets freq table fr tokens in all docs(comments)-Hence lemmatization applied
tfidf = TfidfTransformer()
clf = RandomForestClassifier()


# train classifier
X_train_counts = vect.fit_transform(X_train)
X_train_tfidf = tfidf.fit_transform(X_train_counts)
clf.fit(X_train_tfidf, y_train)  # fitting alg with vectors(by applyng countvectorization and tfidf)


# predict on test data
X_test_counts = vect.transform(X_test)
X_test_tfidf = tfidf.transform(X_test_counts)
y_pred = clf.predict(X_test_tfidf)


pickle.dump(vect, open('transform.pkl', 'wb'))
pickle.dump(tfidf, open('transform1.pkl', 'wb'))


# test accuracy
print("Test accuracy")
print(clf.score(X_test_tfidf, y_test)*100)


x1 = vect.transform(X)
x1_tfidf = tfidf.transform(x1)
print(clf.score(x1_tfidf, y)*100)


pickle.dump(clf, open('model.pkl', 'wb'))
print('Success')
