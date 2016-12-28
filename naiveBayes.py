import nltk
import numpy as np
from sklearn.naive_bayes import GaussianNB
import pandas as pd


letters = pd.read_csv('letter-recognition.txt')

training_points = np.array(letters[:15000].drop(['letter'], 1))
training_labels = np.array(letters[:15000]['letter'])


clf = GaussianNB()
clf.fit(training_points, training_labels)


test_points = np.array(letters[15000:].drop(['letter'], 1))
test_labels = np.array(letters[15000:]['letter'])
# predicts = clf.predict(test_points)

accuracy = clf.score(test_points, test_labels)

print(float(accuracy))


