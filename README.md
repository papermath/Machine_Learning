# Machine Learning

This repository is just a collection of my path on becoming a professional machine learning specialist.

Modules I will be using are
* numpy
* pandas
* re
* sklearn
* matplotlib
* seaborn

## amazon_review.py 
I only used the test.csv of [Amazon Reviews](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews?select=test.csv) due to the size of the csv file. I used naive bayes for this dataset.
## number_recognition.py 
I used the [MNIST numbers](https://www.kaggle.com/competitions/digit-recognizer) dataset. I also use Support Vector Machine which utilizes less code and better accuracy, but has the downside of being computationhally expensive. The SVM has 96% accuracy but could probably be higher because I was only able to tune a couple hyperparameters. Currently the modules matplotlib and seaborn are unused in the two python files, but I will probably add a 2D and 3D plot of the clusters by reducing the amount of features with PCA or LDA. 
## raisin.py
[raisin dataset](https://www.kaggle.com/datasets/muratkokludataset/raisin-dataset) I used SVM and Logistic Regression with Logistic Regression performing far better than SVM partly because I didn't tuned the SVM's hyperparameters as much as I did with Logistic Regression. 
## pumpkin.py
[pumpkin dataset](https://www.kaggle.com/datasets/muratkokludataset/pumpkin-seeds-dataset) achieved a max of 88% accuracy on the test data. I used logistic regression, random forests, adaptive boosting, and gradient boosting. After roughly tuning the hyperparameters I can conclude that they all achieve roughly the same accucary. The output of the code looks something like this
```
Stacked Logistic Regression, Random Forest, Adaptive Boost, and Gradient Boosting Classifier Score:
0.876
Logistic Regression Classifier Score:
0.874
Random Forest Classifier Score: 
0.882
Adaptive Boosting Classifier Score:
0.872
Gradient Boosting Classifier Score:
0.884
```
## bank.py 
This uses a [bank marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) dataset from UCI Machine Learning and utilizes a simple Pipleine with two main models in the search space, a logistic regression and a support vector classifier. I was not able to test it due to the complexity of the support vector classifier, but I was able to conclude the logistic regression obtained around 90% accuracy.
