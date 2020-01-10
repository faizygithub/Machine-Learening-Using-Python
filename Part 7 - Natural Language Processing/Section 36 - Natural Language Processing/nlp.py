# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
#review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])
#review = review.lower()
#review = review.split()
# Next sterp in cleaning is stemming for sparcity e:g loved ,love ->love
#ps=PorterStemmer()
#review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # to make list word again string
#review=' '.join(review)
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    # Next sterp in cleaning is stemming for sparcity e:g loved ,love ->love
    ps=PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # to make list word again string
    review=' '.join(review)
    corpus.append(review)
    
# Creating Bag or wrod model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)    
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()
#classifier.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=200,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


    
    