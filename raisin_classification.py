import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC

raisin = pd.read_csv("Raisin_Dataset.csv")

raisin.Class = raisin.Class.map({"Kecimen": 0, "Besni": 1})

x = raisin.drop(columns = ["Class"])
y = raisin.Class

scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x), columns = x.columns)

for i in [4,5,6]:
    pca = PCA(n_components = i)
    x = pca.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state = 100)

    svm = SVC(kernel = "rbf", random_state = 100)

    parameters = {"gamma": np.logspace(-20,-1,40), "C": np.logspace(-20,-1,40)}

    grid = GridSearchCV(estimator = svm, param_grid = parameters, cv = 5)
    grid.fit(x_train, y_train)

    print(i)
    print(grid.best_estimator_)
    print(grid.score(x_test,y_test))

#SVC(C=0.032570206556597696, gamma=0.1, random_state=100)
#0.8555555555555555


