#currently working on branch logistic
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV

raisin = pd.read_csv("Raisin_Dataset.csv")

raisin.Class = raisin.Class.map({"Kecimen": 0, "Besni": 1})

x = raisin.drop(columns = ["Class"])
y = raisin.Class

scaler = StandardScaler()
scaled_x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(scaled_x, y, train_size = 0.8, random_state = 42)

lr = LogisticRegression(random_state = 42, C = 0.01, tol = 0.00001, max_iter = 10000, solver = "newton-cholesky")

lr.fit(x_train, y_train)

print(lr.score(x_train, y_train))
print(lr.score(x_test, y_test))