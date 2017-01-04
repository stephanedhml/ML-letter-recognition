import numpy as np
from sklearn.svm import SVC
import pandas as pd
from sklearn import metrics

letters = pd.read_csv('letter-recognition.txt')

training_points = np.array(letters[:15000].drop(['letter'], 1))
training_labels = np.array(letters[:15000]['letter'])

clf = SVC()
clf.fit(training_points, training_labels) 


test_points = np.array(letters[15000:].drop(['letter'], 1))
test_labels = np.array(letters[15000:]['letter'])

expected = test_labels
predicted = clf.predict(test_points)

accuracy = clf.score(test_points, test_labels)

print(float(accuracy))

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))