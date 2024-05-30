# Comparitive Analysis of Supervised Machine Learning Algorithms:

## Abstract
This study evaluates three supervised learning algorithms—NaiveBayes Classifier, Random Forest, and Logistic Regression—on a binary classification task to elucidate their performance metrics and applicability. Utilizing Python libraries, we undertook a rigorous analysis involving dataset exploration, model training, hyperparameter tuning via K-Fold CrossValidation and GridSearchCV, and performance evaluation using metrics such as precision, recall, F1-Score, and accuracy. Our findings, presented through confusion matrices,classification reports and Precision-Recall curves, reveal insightful distinctions in algorithm efficacy, contributing to a deeper understanding of their strengths and weaknesses in binary classification scenarios.

**Key words**: NaiveBayesClassifer , RandomForestClassifier , LogisticRegression , F1-score ,Accuracy , Precision , KFold, Cross Validation , GridSearchCV

### INTRODUCTION
Classification algorithms serve as the cornerstone of supervised machine learning, enabling the categorization of data into predefined classes based on its attributes. Their applications span from email filtering and speech recognition to medical diagnosis, showcasing their versatility and critical role in driving insights and automation across various domains. In this study, we perform a binary classification task on the dataset dataset *assignment1.csv* provided to us as part of the study.

#### 1.1 Imported Packages
• **Data Manipulation and Visualisation:** pandas, NumPy, Matplotlib, seaborn

• **Machine Learning models:** Scikit-learn (for model building, hyperparameter tuning, and evaluation)

#### 1.2 Development Process

##### 1.2.1 Loading the dataset
We loaded the dataset using the pd.read csv() from pandas command and named it as df.
##### 1.2.2 Exploratory Data Analysis
In this phase of our analysis, we conducted an in-depth examination of our dataset to uncover key insights. Utilizing a suite of commands like df.info() and df.describe() from the pandas library, we extracted essential statistics that provided a comprehensive view of our data. Our dataset consists of 700 entries, each with 10 features, with values ranging from 1 to 10, indicating a well-structured and diverse dataset. Visualization techniques, employing the matplotlib and seaborn libraries, allowed us to explore the distribution and relationship between various features. This visual exploration revealed that the distributions of values across different classes significantly diverge for numerous features, notably *feature2, feature3, feature4, feature5, feature7, and feature8.* The pronounced differences in their median values between classes underscore their potential as key discriminators for class prediction. Additionally, our analysis identified outliers and varying variances across features within each class, which could influence the model’s performance.

An analysis of class distribution indicated a slight imbalance, with 65.6% of entries belonging to class 0 and 34.4% to class 1. While such imbalances often necessitate the application of resampling techniques like SMOTE, RandomOverSampler, or RandomUnderSampler, our evaluation determined that the current level of imbalance did not warrant their use, as these methods are typically reserved for cases of extreme imbalance. A particularly noteworthy finding from our correlation analysis is the strong positive correlation of features 2, 3, and 6 with the class label, highlighting their importance as predictors. This insight into the dataset’s characteristics is instrumental in guiding our subsequent model selection and training processes, ensuring that we capitalize on the most informative features for classification.

##### 1.2.3 Data Preprocessing

Before proceeding to the model building process, it was important for us to identify our target variable which is the class column in our case. Then we proceeded to split the data into Training, Validation and Test data. To achieve a balanced distribution, we allocated
Figure 3. Heat-map showing the correlation between the various feautures Figure 4. Percentage of class distribution 60% of our dataset to training purposes, allowing our models to learn from a comprehensive range of data points and patterns. The remaining 40% was equally divided between validation and testing, each receiving 20% of the overall data. This split ensured that we had a substantial portion of data for fine-tuning the models during the validation phase and a separate, untouched segment for the final evaluation during testing.

##### 1.2.4 Determining our Evaluation Metric
Another important aspect for us was to determine our primary evaluation metric for our models. Given the slight imbalance in our dataset, we recognized that accuracy might not be the most suitable evaluation metric as it could be biased towards the majority class.
A more appropriate metric for our needs is the f1-macro score, which averages the F1 scores computed for each class independently, thereby giving equal weight to each class irrespective of its size. The f1-macro score is particularly useful in situations like our where class imbalance is present, the F1-Macro score is calculated as follows:

\[ F1_{\text{Macro}} = \frac{1}{N} \sum_{i=1}^{N} F1_i \]

where \( F1_i \) is the F1 score for class \( i \), and \( N \) is the total number of classes. The F1 score for each class is the harmonic mean of precision and recall for that class, providing a balance between the precision and recall metrics.

\[ F1_{\text{Micro}} = \frac{2 \cdot \text{Precision}_{\text{micro}} \times \text{Recall}_{\text{micro}}}{\text{Precision}_{\text{micro}} + \text{Recall}_{\text{micro}}} \]

Additionally, we developed several functions to assist in evaluating our models more effectively. The evaluation matrix function is used to plot the confusion matrix and generate the classification report, providing insights into the precision, recall, and F1 score for each class. Furthermore, the plot precision-recall curve function helps visualize the trade-off between precision and recall for our models, offering a graphical representation of their performance across different thresholds.

##### 1.2.5 Model Building
For our model building, we used the following classifiers to classifiers • NaiveBayes Classifier: NaiveBayesClassifier is a very fast and efficient classifier making it ideal to perform classification tasks due to its assumption of feature independence.

• **LogisticRegression:** LogisticRegression offers straightforward interpretation of feature importance and relationships, making it valuable for understanding the influence of individual features on the outcome and it can effectively handle linearly separable data
with a good balance between precision and recall.

• **RandomForestClassifier:** it is robust to overfitting and provides insights to feature importance.

##### 1.2.6 Hyperparameter Tuning
We used a combination of KFold Cross Validation and GridSearchCV for hyperparameter tuning of our various models. Let us see how we performed it in our models.

• **NaiveBayes Classifier:** For the Naive Bayes classifier we
used the GridSearchCV method combined with KFold crossvalidation to optimize the var smoothing parameter, which is crucial for model performance. The param grid was defined to explore a range of var smoothing values, spread logarithmically between 1 and 10 −9 , comprising 100 values in total. The KFold method is set to split the dataset into 5 folds, with shuffling enabledand a fixed random state to ensure reproducibility. This setup is then passed into GridSearchCV, along with the Naive Bayes model and the specified scoring metric, f1 macro, which evaluates the model’s harmonic mean of precision and recall, averaged across  classes, making it suitable for imbalanced datasets. Upon fitting GridSearchCV to the training data, the best performing model and its parameters are extracted. The optimal var smoothing value is reported, and the optimized model is then used to make predictions on the training data.

• **LogisticRegression:** We use the same methodology here. The hyperparameter space explored includes a variety of regularization strengths (C) ranging logarithmically from 0.001 to 100, types of regularization (penalty) including L1 and L2, and the liblinear
solver that is adept at handling L1 penalties. KFold is employed to split the dataset into 5 parts with shuffling to ensure data diversity across folds, using a consistent random state for reproducibility. This setup is fed into GridSearchCV, specifying the Logistic
Regression model (lr), the defined parameter grid (param grid), the cross-validation strategy (cv=kfold), and using accuracy as the scoring metric. After executing the grid search on the training data, the configuration yielding the highest accuracy is identified, and the best parameters are disclosed. The Logistic Regression model, optimized with these parameters, is then applied to the training data to generate predictions.

• **RandomForestClassifier:** The parameter grid is thoughtfully chosen to cover a broad spectrum of critical hyperparameters for Random Forest, including the number of trees (n estimators), tree depth (max depth), and criteria for node splitting (min samples split, min samples leaf), alongside the choice of using bootstrap samples. By employing a KFold strategy with 5 folds and shuffling, the dataset is meticulously divided to ensure model robustness and generalizability, backed by a stable random state for consistent splits across runs.

Upon fitting the grid search to the training data, GridSearchCV diligently explores the parameter space, evaluating the performance of each configuration based on accuracy, thereby aiming to pinpoint the optimal setup for the Random Forest model on the given
dataset. Once the best parameters are identified, they are applied to instantiate an optimized version of the Random Forest classifier, which is then trained and used to predict outcomes on the training data.


### EVALUATION:

As explained earlier, we are choosing the f1-macro as the primary  evaluation metric for determining the best classification algorithm for us. Let us discuss the results of each classification algorithm in more detail.

**2.1 NaiveBayes**
We get the best performance on our test data for NaiveBayes as seen in Figure 5 with a f1-marco score of 0.98321. Our model performed
better on test data than training data which had a f1-macro score of
0.948 and Validation data which had a score of 0.953. We got a
score of 0.961 on the training set after tuning our parameters using
KFold and GridSearchCV. Our model had a perfect recall of 1 for
class 0 and perfect precision for class 1 and also an accuracy of 0.99
which suggests our model performed exceptionally well on unseen
data.
The matrix in Figure 5 shows that out of all the test samples,
96 were correctly predicted as class 0 (true negatives), and 42 were
correctly predicted as class 1 (true positives). There were no instances where class 0 was incorrectly predicted as class 1 (false
positives), which is ideal. However, there were 2 instances where
the model incorrectly predicted class 1 as class 0 (false negatives).
This indicates that while the model is very accurate in predicting
the negative class, there is a small margin of error in identifying the
positive class.
The Precision-Recall curve in Figure 5 shows a very high level
of performance from our model. The curve starts offs with a precision and recall of 1 and has only a slight dip; this means that the
model was able to retrieve all the relevant instances (high recall)
while also being correct in its retrieval (high precision) for most of
its predictions.

**2.2 LogisticRegression**

For Logistic Regression, the confusion matrix demonstrates outstanding model performance on the test data, akin to the Naive
Bayes results. It correctly classified 96 instances as class 0 and 42
as class 1, while only misclassifying 2 instances of class 1 as class
0, leading to a high f1-macro score, indicative of excellent precision and recall balance. The classification report mirrors this with
perfect precision for class 0, high recall for class 1, and an overall
accuracy of 0.99 and a f1-macro score of 0.98321 which can be
seen in Figure 6

**2.3 RandomForestClassifier**
For the Random Forest classifier, the model displayed a high level
of accuracy on the test data as well. The confusion matrix in Figure 7 shows 92 true negatives and 43 true positives, with slight
misclassifications reflected by 4 false positives and 1 false negative.
The f1-macro score of approximately 0.959 suggests a very effective model, although slightly less precise than the Naive Bayes and
Logistic Regression models, with the precision-recall curve reinforcing the model’s strong performance.

#### CONCLUSION 

In our comprehensive analysis, from 1 it is evident that the Naive
Bayes Classifier and Logistic Regression algorithms exhibited exceptional performance on our test dataset, post the meticulous
process of hyperparameter tuning. This superior performance can
largely be attributed to the underlying assumption of feature independence integral to both algorithms. Conversely, while the RandomForest Classifier also demonstrated commendable efficacy, a
noticeable decline in performance was observed. This decline can
likely be ascribed to the inherent characteristics of the RandomForest Classifier’s mechanism, which, in its attempt to closely learn
from the training data, may have compromised its ability to generalize effectively, thereby affecting its performance on unseen data.
This project highlighted the critical aspects of supervised
learning, particularly hyperparameter optimization’s impact on
model performance. Future efforts may delve into further refining
model parameters, employing advanced ensemble techniques, and
exploring new feature selection methods to boost accuracy and robustness. This assignment has reinforced the value of precision in
machine learning workflows, explaining how important it is in realworld scenarios.
