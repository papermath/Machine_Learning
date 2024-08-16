import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

reviews = pd.read_csv("amazon_reviews.csv")

reviews.dropna(inplace = True)

reviews.like  = reviews.apply(lambda row: row.like - 1, axis = 1)
y = reviews.like
x = reviews.description
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8, test_size = 0.2, random_state = 100)

count = CountVectorizer()
count.fit(x_train + x_test)
x_counts = count.transform(x_train)
print(x_counts)
