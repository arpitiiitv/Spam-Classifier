from flask import Flask, render_template, request
import pickle
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

app = Flask(__name__)
filename = 'spam_detecter.pkl'
with open(filename, 'rb') as f:
    spam_detector = pickle.load(f)
filename = 'cv.pkl'
with open(filename,'rb') as f:
    cv = pickle.load(f)

def clear_sent(sms_data):
    stemmer = PorterStemmer()
    corpus = []
    # replace non-alphabates with space
    row_data = re.sub('[^a-zA-Z]', " ", sms_data)
    # convert words into lowercase
    row_data = row_data.lower()
    # make list from sentences
    row_data_list = row_data.split()
    # remove stopwords
    important_row_data = [stemmer.stem(word) for word in row_data_list if word not in set(stopwords.words('english'))]
    data = ' '.join(important_row_data)
    # append to corpus
    corpus.append(data)
    return corpus

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/detection',methods=['POST'])
def detection():
    sms_data = request.form['message']
    data = clear_sent(sms_data)
    result = ""
    if len(sms_data)==0:
        result = "Please enter valid string".upper()
    else:
        vect = cv.transform(data).toarray()

        if spam_detector.predict(vect)==0:
            result = "This message is not spam".upper()
        else:
            result = "This message is spam".upper()
    return render_template('result.html',result=result)


if __name__ == '__main__':
    app.run(debug=True)