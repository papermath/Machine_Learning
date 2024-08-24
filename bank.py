import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

bank = pd.read_csv("bank-full.csv", delimiter = ";")
bank.pdays = bank.pdays.replace(-1,0)
bank.pdays = bank.pdays.infer_objects(copy = False)

bank.default = bank.default.replace({"no":0, "yes":1})
bank.default = bank.default.infer_objects(copy = False)

bank.housing = bank.housing.replace({"no":0, "yes":1})
bank.housing = bank.housing.infer_objects(copy = False)

bank.loan = bank.loan.replace({"no":0, "yes":1})
bank.loan = bank.loan.infer_objects(copy = False)

bank.y = bank.y.replace({"no":0, "yes":1})
bank.y = bank.y.infer_objects(copy = False)

x = bank.drop(columns = ["y"])
y = bank.y

num_cols = []
cat_cols = []
for name,dtype in x.dtypes.items():
    if dtype == "int64":
        num_cols.append(name)
    if dtype == "object":
        cat_cols.append(name)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 69)
numerical = Pipeline([("scaler", StandardScaler())])
categorical = Pipeline([("ohe", OneHotEncoder(categories = "auto", sparse_output = False,drop = "first"))])
preprocess = ColumnTransformer([("num", numerical, num_cols), ("cat", categorical, cat_cols)])
model = Pipeline([("preprocess", preprocess), ("pca", PCA()), ("model", LogisticRegression())])

search_space = [{"model": [LogisticRegression(solver = "liblinear")], "model__penalty": ["l1","l2"], "pca__n_components": np.linspace(30,40).astype(int)}, {"model": [SVC(kernel = "rbf")], "model__C": [0.1,1,10], "model__gamma": [0.1,1,10], "pca__no_components": np.linspace(30,40).astype(int)}]
grid = GridSearchCV(estimator = model, param_grid = search_space, cv = 5)

grid.fit(x_train, y_train)
print(grid.best_estimator_)
print(grid.best_params_)
print(grid.score(x_test, y_test))
