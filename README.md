# Machine Learning

This repository is just a collection of my path on becoming a professional machine learning specialist.

Modules I will be using are
* numpy
* pandas
* sklearn
* matplotlib
* seaborn

For amazon_review.py I only used the test.csv of [Amazon Reviews](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews?select=test.csv) due to the size of the csv file. I used naive bayes for this dataset. For number_recognition.py I used the [MNIST numbers](https://www.kaggle.com/competitions/digit-recognizer) dataset. I also use Support Vector Machine which utilizes less code and better accuracy, but has the downside of being computationhally expensive. The SVM has 96% accuracy but could probably be higher because I was only able to tune a couple hyperparameters. Currently the modules matplotlib and seaborn are unused in the two python files, but I will probably add a 2D and 3D plot of the clusters by reducing the amount of features with PCA or LDA. For the [raisin dataset](https://www.kaggle.com/datasets/muratkokludataset/raisin-dataset) I used SVM and Logistic Regression with Logistic Regression performing far better than SVM partly because I didn't tuned the SVM's hyperparameters as much as I did with Logistic Regression. The [pumpkin dataset](https://www.kaggle.com/datasets/muratkokludataset/pumpkin-seeds-dataset) achieved a max of 88% accuracy on the test data. I used logistic regression, random forests, adaptive boosting, and gradient boosting. After roughly tuning the hyperparameters I can conclude that they all achieve roughly the same accucary. The output of the code looks something like this
`
Logistic Regression Classifier Score:<br>
0.874  
Random Forest Classifier Score:  
0.882  
Adaptive Boosting Classifier Score:  
0.872  
Gradient Boosting Classifier Score:  
0.884  
`


