Recruitment Task – Classification of a Heart with Hypertrophic Cardiomyopathy (Cardiomegaly)

This is my model which detects if patiens has Cardiomegaly or not.
It bases on the example which you attached.
I used all five methods of machine learning. 

KNN methon:
Cross-validation mean score: 0.824
Standard deviation of CV score: 0.012
But on testing data:
Accuracy on test data: 0.625 
The model performs reasonably, but the test accuracy is not very high.

DECISION TREE method:
Cross-validation mean score: 0.786
Standard deviation of CV score: 0.140
Accuracy on test data: 0.5
Theese are not the best results, the model overfits the training data and does not generalize well..

RANDOM FOREST method:
Cross-validation mean score: 0.898
Standard deviation of CV score: 0.083
Accuracy on test data: 0.625
The model shows better training performance, but test accuracy did not improve compared to KNN.

Logistic Regression, which turned out to be the best method:
Accuracy on test data: 0.875
Which is really satisfying result.

SVC:
Accuracy on test data: 0.75
Good try, but slightly lower than Logistic Regression.

final comparison:
KNN: Accuracy = 0.625
Decision Tree: Accuracy = 0.500
Random Forest: Accuracy = 0.625
Logistic Regression: Accuracy = 0.875
SVC: Accuracy = 0.750

Conclusion: Logistic Regression is the most effective model for this dataset, achieving the highest accuracy on test data.
It is not a perfect score but I didn't have a very large data set.