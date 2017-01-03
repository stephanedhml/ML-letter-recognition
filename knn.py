import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

letters = pd.read_csv('letter-recognition.txt')

training_points = np.array(letters[:15000].drop(['letter'], 1))
training_labels = np.array(letters[:15000]['letter'])

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(training_points, training_labels) 


test_points = np.array(letters[15000:].drop(['letter'], 1))
test_labels = np.array(letters[15000:]['letter'])


accuracy = neigh.score(test_points, test_labels)
print(float(accuracy))