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
x_train_list = list(x_train)
x_test_list = list(x_test)

count = CountVectorizer()
count.fit(x_train_list + x_test_list)
x_train_counts = count.transform(x_train_list)
x_test_counts = count.transform(x_test_list)

bayes = MultinomialNB()
bayes.fit(x_train_counts, y_train)

review = count.transform(["I love the style of this, but after a couple years, the DVD is giving me problems. It doesn't even work anymore and I use my broken PS2 Now. I wouldn't recommend this, I'm just going to upgrade to a recorder now. I wish it would work but I guess i'm giving up on JVC. I really did like this one... before it stopped working. The dvd player gave me problems probably after a year of having it."])
prediction = bayes.predict_proba(review)
print(prediction)


