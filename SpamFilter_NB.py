import os
import io
import numpy
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import time
from nltk import word_tokenize
#from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

stemmer = PorterStemmer()
vectorInst = CountVectorizer()
analyzer = vectorInst.build_analyzer()

#function to use PorterStemmer from NTLK on each word in mail sample
def stemmed_words(doc):
    return [stemmer.stem(t) for t in analyzer(doc)]

def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    print(len(rows)," files")
    return DataFrame(rows, index=index)

start = time.time()
data = DataFrame({'message': [], 'class': []})

print("Started reading spam .. ")
data = data.append(dataFrameFromDirectory('/home/rakesh/Downloads/MachineLearning/Spam', 'spam'))

print("Now Started reading Ham .. ")
data = data.append(dataFrameFromDirectory('/home/rakesh/Downloads/MachineLearning/Ham', 'ham'))

print("Count vectorizing data.. ")

#vectorizer = CountVectorizer(analyzer=stemmed_words)
vectorizer = CountVectorizer() # This will use inbuilt tokenizer without applying lemmatizer / porter stemmer
counts = vectorizer.fit_transform(data['message'].values)
print("Vectorizer output:", str(counts.shape))

end = time.time()
print("Time to read and vectorize is:", end-start)

start1 = time.time()
classifier = MultinomialNB()

#convert to encoded integers, since it makes us easy to easy sklearn metrics for deriving prediction metrics later
targets_enc = pd.factorize(data['class'])
end1 = time.time()

print("Time to fit using Multinomial Naive Bayes is:", end-start)

#classifier.fit(counts, targets) -- we could actually directly fit with labels data without having to convert
X_train, X_test, y_train, y_test = train_test_split( counts, targets_enc[0], test_size=0.25)
classifier.fit(X_train, y_train)

#examples = ['Free Viagra now!!!', "Hi Bob, how about a game of golf tomorrow?"]
#example_counts = vectorizer.transform(examples)
y_predict = classifier.predict(X_test)
end2  = time.time()

#Compute prediction metrics using sklearn metrics
print("Overall time:", end2-start)
print("Precision:", precision_score(y_test,y_predict))
print("Accuracy:", accuracy_score(y_test, y_predict))
print("F1-Score:", f1_score(y_test, y_predict))
