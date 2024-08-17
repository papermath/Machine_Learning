import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn

numbers = pd.read_csv("numbers.csv")

x = numbers.drop(columns = ["label"])
y = numbers.label

scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components = 400, random_state = 100)
x = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8, random_state = 100)

svm = SVC(kernel = "rbf")
svm.fit(x_train, y_train)
print(svm.score(x_test, y_test))
#0.9519047619047619 acc with PCA before scaling
#0.9605952380952381 acc with PCA after scaling
#0.8653571428571428 acc with C = 0.1
#0.9569047619047619 acc with C = 10

