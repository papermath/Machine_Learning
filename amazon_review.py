import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

print("Input your review here and watch the machine classify if your review was positive or negative!")
title = input("Title: ")
description = input ("Description: ")

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

review = [title, description]

review_count_description = count.transform([review[1]])

bayes = MultinomialNB()
bayes.fit(x_train_counts, y_train)

description_proba = bayes.predict_proba(review_count_description)[0][1]

p = reviews.title
q = reviews.like

p_train, p_test, q_train, q_test = train_test_split(p,q,train_size = 0.8, test_size = 0.2, random_state = 100)

p_train_list = list(p_train)
p_test_list = list(p_test)

count_2 = CountVectorizer()
count_2.fit(p_train_list + p_test_list)
p_train_count = count_2.transform(p_train_list)
p_test_count = count_2.transform(p_test_list)

bayes_2 = MultinomialNB()

bayes_2.fit(p_train_count, q_train)

review_count_title = count_2.transform([review[0]])
title_proba = bayes_2.predict_proba(review_count_title)[0][1]

x_test_count = count.transform(x_test)
p_test_count = count_2.transform(p_test)
x_test_proba = bayes.predict_proba(x_test_count)
p_test_proba = bayes_2.predict_proba(p_test_count)

x_pos_test_proba = [i[1] for i in x_test_proba]
p_pos_test_proba = [i[1] for i in p_test_proba]
df = pd.DataFrame({"title_proba": p_pos_test_proba, "description_proba": x_pos_test_proba, "like": y_test})
reg = LogisticRegression()
reg.fit(df[["title_proba","description_proba"]], df["like"])
predict_proba = reg.predict_proba(np.array([title_proba, description_proba]).reshape(1,-1))
if reg.predict(np.array([title_proba, description_proba]).reshape(1,-1)) == [0]:
    print("The machine is " + str(predict_proba[0][0]) + f"% sure the review was negative.")
elif reg.predict(np.array([title_proba, description_proba]).reshape(1,-1)) == [1]:
    print("The machine is " + str(predict_proba[0][1]) + f"% sure the review was positive.")
else:
    print("ERROR can't determine.")
