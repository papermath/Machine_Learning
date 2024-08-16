import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

reviews = pd.read_csv("amazon_reviews.csv")

reviews.dropna(inplace = True)

reviews.like  = reviews.apply(lambda row: row.like - 1, axis = 1)

x = reviews.title
y = reviews.like

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8, test_size = 0.2, random_state = 100)

x_train_list = list(x_train)
x_test_list = list(x_test)
print("stop")
count = CountVectorizer()
count.fit(x_train_list + x_test_list)
x_train_count = count.transform(x_train_list).toarray()
x_test_count = count.transform(x_test_list).toarray()
print("stop")
bayes = MultinomialNB()

bayes.fit(x_train_count, list(y_train))
print("stop")
print(bayes.score(x_test_count, y_test))


