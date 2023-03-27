from flask import Flask, request, render_template
import pickle
import pandas as pd
df = pd.read_csv("Tweets.csv")

data = 'model.pkl'
model = pickle.load(open(data, 'rb'))
tfidf = pickle.load(open('transform1.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    comment = [x for x in request.form.values()]
    print(comment)

    x = df.selected_text.values
    x = vect.transform(comment)
    x_tfidf = tfidf.transform(x)

    o = model.predict(x_tfidf)
    print(o)

    x_prob = model.predict_proba(x_tfidf)
    x_prob = '{0:.{1}f}'.format(x_prob[0][1], 2)
    print(x_prob)

    if o[0] == 0:
        return render_template('index.html', prog='It is a Negative Tweet', prob='{}'.format(x_prob))
    elif o[0] == 1:
        return render_template('index.html', prog='It is a Neutral Tweet', prob='{}'.format(x_prob))
    else:
        return render_template('index.html', prog='It is a Positive Tweet', prob='{}'.format(x_prob))


if __name__ == '__main__':
    app.run(debug=True)
