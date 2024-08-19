import pandas as pd
import numpy as np
from scipy.stats import uniform
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA

raisin = pd.read_csv("Raisin_Dataset.csv")

raisin.Class = raisin.Class.map({"Kecimen":0, "Besni": 1})

x = raisin.drop(columns = ["Class"])
y = raisin.Class

scaler = StandardScaler()
scaled_x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(scaled_x, y, train_size = 0.8, random_state = 69)

svm = SVC(kernel = "rbf", C = 1, gamma = 0.01, random_state = 69)
svm.fit(x_train, y_train)
#GridSearchCV outputted C = 1 and gamma = 0.01 as best values with train score of 0.8819444444444444 and test score of 0.8166666666666667
#RandomSearchCV outputted C = 1 and gamma = 0.07422867291585378 as best values with train score of 0.8847222222222222 and test score of 0.8277777777777777
#PCA decreased accuracy by around half
#but due to how close they are I will just let C = 1 and gamma = 0.01 for simplicity

print(svm.score(x_test, y_test))


