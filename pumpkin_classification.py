import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

pumpkin = pd.read_csv("Pumpkin_Seeds_Dataset.csv")

pumpkin.Class = pumpkin.Class.map({"Çerçevelik": 0, "Ürgüp Sivrisi": 1})

x = pumpkin.drop(columns = ["Class"])
y = pumpkin.Class

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state = 69)

lr = LogisticRegression(solver = "liblinear", C = 1, penalty = "l2")
lr.fit(x_train, y_train)
print("Logistic Regression Classifier Score:")
print(lr.score(x_test, y_test))

rfc = RandomForestClassifier(max_depth = 7, max_features = 8)
rfc.fit(x_train, y_train)
print("Random Forest Classifier Score:")
print(rfc.score(x_test, y_test))

tree_stump = DecisionTreeClassifier(max_depth = 1)
ada = AdaBoostClassifier(estimator = tree_stump, learning_rate = 0.25, n_estimators = 250, algorithm = "SAMME", random_state = 69)
ada.fit(x_train, y_train)
print("Adaptive Boosting Classifier Score:")
print(ada.score(x_test, y_test))

gradient = GradientBoostingClassifier(n_estimators = 50)
gradient.fit(x_train, y_train)
print("Gradient Boosting Classifier Score:")
print(gradient.score(x_test, y_test))


