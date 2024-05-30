# Comparitive Analysis of Supervised Machine Learning Algorithms:

## Abstract
SUMMARY
This study evaluates three supervised learning algorithms—NaiveBayes Classifier, Random Forest, and Logistic Regression—on a binary classification task to elucidate their performance metrics and applicability. Utilizing Python libraries, we undertook a rigorous analysis involving dataset exploration, model training, hyperparameter tuning via K-Fold CrossValidation and GridSearchCV, and performance evaluation using metrics such as precision, recall, F1-Score, and accuracy. Our findings, presented through confusion matrices,classification reports and Precision-Recall curves, reveal insightful distinctions in algorithm efficacy, contributing to a deeper understanding of their strengths and weaknesses in binary classification scenarios.

**Key words**: NaiveBayesClassifer , RandomForestClassifier , LogisticRegression , F1-score ,Accuracy , Precision , KFold, Cross Validation , GridSearchCV

###INTRODUCTION
Classification algorithms serve as the cornerstone of supervised machine learning, enabling the categorization of data into predefined classes based on its attributes. Their applications span from email filtering and speech recognition to medical diagnosis, showcasing their versatility and critical role in driving insights and automation across various domains. In this study, we perform a binary classification task on the dataset dataset *assignment1.csv* provided to us as part of the study.

#### 1.1 Imported Packages
• **Data Manipulation and Visualisation:** pandas, NumPy, Matplotlib, seaborn

• **Machine Learning models:** Scikit-learn (for model building, hyperparameter tuning, and evaluation)

#### 1.2 Development Process

##### 1.2.1 Loading the dataset
We loaded the dataset using the pd.read csv() from pandas command and named it as df.
##### 1.2.2 Exploratory Data Analysis
In this phase of our analysis, we conducted an in-depth examination of our dataset to uncover key insights. Utilizing a suite of commands like df.info() and df.describe() from the pandas library, we
extracted essential statistics that provided a comprehensive view of
