#iniziato giorno: 13/03/2025
#finito giorno: 15/03/2025

#import di librerie varie
import os
import pickle
from unicodedata import category
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#preparazione dati
#C:/Users/drj4k/OneDrive/Desktop/AI_projects_Fantini/progettoParcheggi/foto_progetto/clf-data -> casa
#C:/Users/fantini.alessandro/Desktop/AI_projects_Fantini/foto_progetto/clf-data -> lab (da modificare path)
input_dir = './progettoParcheggi/foto_progetto/clf-data'
categories = ['empty', 'not_empty']

data = []
labels = []

for category_idx, category in enumerate(categories):
    for files in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, files)
        img = imread(img_path)
        img = resize(img, (15,15))
        data.append(img.flatten())
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

#addestramento / divisione test
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, shuffle = True, stratify = labels)

#addestramento classificatore
classifier = SVC()

parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)

#test
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% campioni classificati correttamente'.format(str(score * 100)))